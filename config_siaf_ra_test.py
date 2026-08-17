"""Sacred configuration for SIAF-RA evaluation."""

import glob
import itertools

import sacred
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False
sacred.SETTINGS.CAPTURE_MODE = "no"

ex = Experiment("SIAF_RA_TEST")
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = [".", "./dataloaders", "./models", "./util"]
sources_to_save = list(
    itertools.chain.from_iterable(
        [glob.glob("%s/*.py" % folder) for folder in source_folders]
    )
)
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    gpu_id = 0
    eval_domains = ["CHAOST2"]
    use_horizontal_flip_tta = False
    reload_model_path = "./runs/SIAF_RA__SABS_1shot/1/snapshots/40000.pth"

    model = {
        "use_coco_init": True,
        "which_model": "dlfcn_res101",
        "siaf_ra": {
            "feat_dim": 256,
            "adapter_hidden": 128,
            "field_build_size": 64,
            "poisson_iterations": 56,
            "harmonic_iterations": 56,
            "directional_kappa": 4.0,
            "field_mask_threshold": 0.30,
            "topk": 12,
            "semantic_temperature": 10.0,
            "affinity_temperature": 0.09,
            "reciprocal_weight": 0.35,
            "coherence_temperature": 0.06,
            "max_field_scale": 2.0,
            "initial_field_scale": 0.60,
            "fg_anchor": 0.70,
            "bg_anchor": 0.30,
            "anchor_softness": 0.08,
            "separation_margin": 0.06,
            "uncertainty_floor": 0.10,
        },
    }

    path = {
        "SABS": {
            "data_dir": "./data/ABD/ABDOMEN_CT/sabs_CT_normalized",
            "test_label": [6, 2, 3, 1],
            "target_size": 256,
        },
        "CHAOST2": {
            "data_dir": "./data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized",
            "test_label": [1, 2, 3, 4],
            "target_size": 257,
        },
        "CARDIAC_bssFP": {
            "data_dir": "./data/Cardiac/bSSFP/cmr_bssFP_normalized",
            "test_label": [1, 2, 3],
            "target_size": 192,
        },
        "CARDIAC_LGE": {
            "data_dir": "./data/Cardiac/LGE/cmr_LGE_normalized",
            "test_label": [1, 2, 3],
            "target_size": 192,
        },
        "Prostate_NCI": {
            "data_dir": "./data/Prostate/NCI/NCI_normalized",
            "test_label": [1, 5, 6],
            "target_size": 192,
        },
        "Prostate_UCLH": {
            "data_dir": "./data/Prostate/UCLH/UCLH_normalized",
            "test_label": [1, 5, 6],
            "target_size": 192,
        },
    }
