#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# SIAF-RA unified train -> immediate test runner.
#
# Default: all six directions, sequentially. Each direction is
# trained first and tested IMMEDIATELY after its 40000.pth exists.
# Only the final checkpoint is saved by train_siaf_ra.py.
#
# Examples:
#   GPU_ID=0 bash run_siaf_ra.sh
#   EXPERIMENTS=CT2MRI GPU_ID=0 bash run_siaf_ra.sh
#   EXPERIMENTS=CT2MRI,MRI2CT GPU_ID=0 bash run_siaf_ra.sh
#   EXPERIMENTS=LGE2bSSFP,bSSFP2LGE GPU_ID=0 bash run_siaf_ra.sh
#
# Resume behavior:
#   SKIP_EXISTING=1 (default) reuses the highest Sacred run id
#   that actually contains the requested final checkpoint.
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/root/rivermind-data/SIAF}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_STEPS="${N_STEPS:-40000}"
USE_TTA="${USE_TTA:-False}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BACKBONE="${BACKBONE:-dlfcn_res101}"
RUN_PREFIX="${RUN_PREFIX:-}"
EXPERIMENTS="${EXPERIMENTS:-ALL}"

TRAIN_PY="${PROJECT_ROOT}/train_siaf_ra.py"
TEST_PY="${PROJECT_ROOT}/test_siaf_ra.py"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/siaf_ra}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/experiment_logs/siaf_ra}"
MASTER_TXT="${RESULT_ROOT}/SIAF_RA_results.txt"
MASTER_TSV="${RESULT_ROOT}/SIAF_RA_summary.tsv"
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

VALID_EXPERIMENTS=(CT2MRI MRI2CT LGE2bSSFP bSSFP2LGE NCI2UCLH UCLH2NCI)
ALL_CSV="CT2MRI,MRI2CT,LGE2bSSFP,bSSFP2LGE,NCI2UCLH,UCLH2NCI"

if [[ "${EXPERIMENTS}" == "ALL" || "${EXPERIMENTS}" == "all" ]]; then
  EXPERIMENTS="${ALL_CSV}"
fi
EXPERIMENTS="$(echo "${EXPERIMENTS}" | tr -d ' ' | sed 's/,,*/,/g; s/^,//; s/,$//')"

contains_csv() {
  local csv="$1" key="$2"
  [[ ",${csv}," == *",${key},"* ]]
}
for item in $(echo "${EXPERIMENTS}" | tr ',' ' '); do
  ok=0
  for valid in "${VALID_EXPERIMENTS[@]}"; do
    [[ "${item}" == "${valid}" ]] && ok=1 && break
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[FATAL] Unknown experiment: ${item}"
    echo "Valid: ${VALID_EXPERIMENTS[*]} or ALL"
    exit 1
  fi
done

[[ -f "${TRAIN_PY}" ]] || { echo "[FATAL] Missing ${TRAIN_PY}"; exit 1; }
[[ -f "${TEST_PY}" ]] || { echo "[FATAL] Missing ${TEST_PY}"; exit 1; }

# ---------------- transfer metadata ----------------
declare -A SOURCE TARGET TRAIN_LABEL TARGET_SIZE
SOURCE[CT2MRI]="SABS";                 TARGET[CT2MRI]="CHAOST2";       TRAIN_LABEL[CT2MRI]="[1,2,3,6]"; TARGET_SIZE[CT2MRI]="257"
SOURCE[MRI2CT]="CHAOST2";              TARGET[MRI2CT]="SABS";          TRAIN_LABEL[MRI2CT]="[1,2,3,4]"; TARGET_SIZE[MRI2CT]="256"
SOURCE[LGE2bSSFP]="CARDIAC_LGE";       TARGET[LGE2bSSFP]="CARDIAC_bssFP"; TRAIN_LABEL[LGE2bSSFP]="[1,2,3]"; TARGET_SIZE[LGE2bSSFP]="192"
SOURCE[bSSFP2LGE]="CARDIAC_bssFP";     TARGET[bSSFP2LGE]="CARDIAC_LGE"; TRAIN_LABEL[bSSFP2LGE]="[1,2,3]"; TARGET_SIZE[bSSFP2LGE]="192"
SOURCE[NCI2UCLH]="Prostate_NCI";       TARGET[NCI2UCLH]="Prostate_UCLH"; TRAIN_LABEL[NCI2UCLH]="[1,5,6]"; TARGET_SIZE[NCI2UCLH]="192"
SOURCE[UCLH2NCI]="Prostate_UCLH";      TARGET[UCLH2NCI]="Prostate_NCI"; TRAIN_LABEL[UCLH2NCI]="[1,5,6]"; TARGET_SIZE[UCLH2NCI]="192"

experiment_dir() {
  local source="$1"
  if [[ -z "${RUN_PREFIX}" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/runs/SIAF_RA__${source}_1shot"
  else
    printf '%s\n' "${PROJECT_ROOT}/runs/SIAF_RA_${RUN_PREFIX}_${source}_1shot"
  fi
}

# Highest numeric Sacred run id with a real final checkpoint wins.
find_latest_final_ckpt() {
  local exp_dir="$1" final_step="$2"
  local best_run=-1 best_ckpt=""
  [[ -d "${exp_dir}" ]] || return 0
  local run_dir run_id ckpt
  shopt -s nullglob
  for run_dir in "${exp_dir}"/*; do
    [[ -d "${run_dir}" ]] || continue
    run_id="$(basename "${run_dir}")"
    [[ "${run_id}" =~ ^[0-9]+$ ]] || continue
    ckpt="${run_dir}/snapshots/${final_step}.pth"
    if [[ -f "${ckpt}" ]] && (( 10#${run_id} > best_run )); then
      best_run=$((10#${run_id}))
      best_ckpt="${ckpt}"
    fi
  done
  shopt -u nullglob
  [[ -n "${best_ckpt}" ]] && printf '%s\n' "${best_ckpt}"
}

touch "${MASTER_TXT}"
if [[ ! -f "${MASTER_TSV}" ]]; then
  printf "Experiment\tSource\tTarget\tMean_DSC\tStd\tN\tCheckpoint\n" > "${MASTER_TSV}"
fi
{
  echo "======================================================================"
  echo "SIAF-RA TRAIN -> IMMEDIATE TEST"
  echo "======================================================================"
  echo "Date        : $(date)"
  echo "Experiments : ${EXPERIMENTS}"
  echo "Steps       : ${N_STEPS}"
  echo "GPU         : ${GPU_ID}"
  echo "Backbone    : ${BACKBONE}"
  echo "TTA         : ${USE_TTA}"
  echo "Resume      : SKIP_EXISTING=${SKIP_EXISTING}"
  echo "Checkpoint  : final ${N_STEPS}.pth only"
  echo "Sizes       : CT2MRI=257, MRI2CT=256, cardiac/prostate=192"
  echo "======================================================================"
} | tee -a "${MASTER_TXT}"

for tag in "${VALID_EXPERIMENTS[@]}"; do
  contains_csv "${EXPERIMENTS}" "${tag}" || continue

  source="${SOURCE[$tag]}"
  target="${TARGET[$tag]}"
  train_label="${TRAIN_LABEL[$tag]}"
  expected_size="${TARGET_SIZE[$tag]}"
  exp_dir="$(experiment_dir "${source}")"
  train_log="${LOG_ROOT}/${tag}__train.log"
  test_log="${LOG_ROOT}/${tag}__test.log"

  echo | tee -a "${MASTER_TXT}"
  echo "########################################################################" | tee -a "${MASTER_TXT}"
  echo "# ${tag}: ${source} -> ${target}" | tee -a "${MASTER_TXT}"
  echo "########################################################################" | tee -a "${MASTER_TXT}"

  ckpt=""
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    ckpt="$(find_latest_final_ckpt "${exp_dir}" "${N_STEPS}" || true)"
  fi

  if [[ -n "${ckpt}" ]]; then
    echo "[skip train] existing final checkpoint: ${ckpt}" | tee -a "${MASTER_TXT}"
  else
    echo "[train] ${tag} starts now" | tee -a "${MASTER_TXT}"
    set +e
    if [[ -z "${RUN_PREFIX}" ]]; then
      "${PYTHON_BIN}" "${TRAIN_PY}" with \
        gpu_id="${GPU_ID}" \
        dataset="${source}" \
        test_label="${train_label}" \
        n_steps="${N_STEPS}" \
        modelname="${BACKBONE}" \
        model.which_model="${BACKBONE}" \
        2>&1 | tee "${train_log}" | tee -a "${MASTER_TXT}"
    else
      "${PYTHON_BIN}" "${TRAIN_PY}" with \
        gpu_id="${GPU_ID}" \
        dataset="${source}" \
        test_label="${train_label}" \
        n_steps="${N_STEPS}" \
        modelname="${BACKBONE}" \
        model.which_model="${BACKBONE}" \
        run_prefix="${RUN_PREFIX}" \
        2>&1 | tee "${train_log}" | tee -a "${MASTER_TXT}"
    fi
    status=${PIPESTATUS[0]}
    set -e
    [[ "${status}" -eq 0 ]] || { echo "[FATAL] training failed: ${tag}"; exit "${status}"; }

    ckpt="$(find_latest_final_ckpt "${exp_dir}" "${N_STEPS}" || true)"
    [[ -n "${ckpt}" ]] || { echo "[FATAL] ${N_STEPS}.pth not found after training ${tag}"; exit 2; }
    echo "[resolved checkpoint] ${ckpt}" | tee -a "${MASTER_TXT}"
  fi

  # IMMEDIATE TEST: do not wait for other directions to train.
  echo "[test] ${tag} starts immediately" | tee -a "${MASTER_TXT}"
  set +e
  "${PYTHON_BIN}" "${TEST_PY}" with \
    gpu_id="${GPU_ID}" \
    reload_model_path="${ckpt}" \
    eval_domains="['${target}']" \
    use_horizontal_flip_tta="${USE_TTA}" \
    model.which_model="${BACKBONE}" \
    2>&1 | tee "${test_log}" | tee -a "${MASTER_TXT}"
  status=${PIPESTATUS[0]}
  set -e
  [[ "${status}" -eq 0 ]] || { echo "[FATAL] testing failed: ${tag}"; exit "${status}"; }

  if ! grep -q "input size    : ${expected_size}x${expected_size}" "${test_log}"; then
    echo "[FATAL] target-size mismatch for ${tag}; expected ${expected_size}x${expected_size}" | tee -a "${MASTER_TXT}"
    grep "input size" "${test_log}" | tee -a "${MASTER_TXT}" || true
    exit 3
  fi

  overall_line="$(grep '^\[OVERALL\]' "${test_log}" | tail -n1 || true)"
  [[ -n "${overall_line}" ]] || { echo "[FATAL] no [OVERALL] summary in ${test_log}"; exit 4; }
  read -r mean std n <<< "$(echo "${overall_line}" | sed -E 's/.*Mean_DSC=([^ ]+) Std=([^ ]+) N=([0-9]+).*/\1 \2 \3/')"

  # Persistent summary: when a selected direction is re-run, replace only
  # that row and keep results from other directions.
  tmp_tsv="${MASTER_TSV}.tmp"
  awk -F '\t' -v tag="${tag}" 'NR==1 || $1 != tag' "${MASTER_TSV}" > "${tmp_tsv}"
  mv "${tmp_tsv}" "${MASTER_TSV}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${source}" "${target}" "${mean}" "${std}" "${n}" "${ckpt}" \
    >> "${MASTER_TSV}"
  {
    echo "[RESULT] ${tag} Mean_DSC=${mean} Std=${std} N=${n}"
    echo "[RESULT] checkpoint=${ckpt}"
  } | tee -a "${MASTER_TXT}"
done

{
  echo
  echo "======================================================================"
  echo "SIAF-RA FINAL SUMMARY"
  echo "======================================================================"
  column -t -s $'\t' "${MASTER_TSV}" 2>/dev/null || cat "${MASTER_TSV}"
  echo "======================================================================"
  echo "TXT: ${MASTER_TXT}"
  echo "TSV: ${MASTER_TSV}"
} | tee -a "${MASTER_TXT}"
