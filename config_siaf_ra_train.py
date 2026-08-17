import glob
import itertools
import os

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False
sacred.SETTINGS.CAPTURE_MODE = "no"

ex = Experiment("SIAF_RA")
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
    seed = 1234
    gpu_id = 0
    num_workers = 8

    dataset = "SABS"
    test_label = [1, 2, 3, 6]
    exclude_label = None
    n_sv = 5000
    min_size = 200
    use_gt = False

    use_coco_init = True
    modelname = "dlfcn_res101"
    batch_size = 1
    ignore_label = 255
    use_wce = True
    freeze_backbone_bn = True

    n_steps = 40000
    max_iters_per_load = 1000
    print_interval = 100

    # Final mask + explicit raw-semantic anchor. There is intentionally no
    # query-field regression loss in SIAF-RA.
    lambda_dice = 0.5
    lambda_semantic_anchor = 0.50
    semantic_anchor_dice = 0.5
    gradient_clip_norm = 5.0

    model = {
        "use_coco_init": use_coco_init,
        "which_model": modelname,
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
            # Reliability-adaptive intervention.
            "max_field_scale": 2.0,
            "initial_field_scale": 0.60,
            "fg_anchor": 0.70,
            "bg_anchor": 0.30,
            "anchor_softness": 0.08,
            "separation_margin": 0.06,
            "uncertainty_floor": 0.10,
        },
    }

    task = {"n_ways": 1, "n_shots": 1, "n_queries": 1}

    optim = {
        "lr": 1e-3,
        "backbone_lr_mult": 0.10,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }

    lr_step_every = 1000
    lr_step_gamma = 0.95
    lr_milestones = list(range(lr_step_every, n_steps + 1, lr_step_every))

    # Leave blank for the default directory SIAF_RA__<dataset>_1shot.
    run_prefix = ""

    path = {
        "log_dir": "./runs",
        "SABS": {"data_dir": "./data/ABD/ABDOMEN_CT/sabs_CT_normalized"},
        "CHAOST2": {"data_dir": "./data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized"},
        "CARDIAC_bssFP": {"data_dir": "./data/Cardiac/bSSFP/cmr_bssFP_normalized"},
        "CARDIAC_LGE": {"data_dir": "./data/Cardiac/LGE/cmr_LGE_normalized"},
        "Prostate_NCI": {"data_dir": "./data/Prostate/NCI/NCI_normalized"},
        "Prostate_UCLH": {"data_dir": "./data/Prostate/UCLH/UCLH_normalized"},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    del command_name, logger
    pieces = [str(config.get("run_prefix", "")), str(config["dataset"]), "%dshot" % int(config["task"]["n_shots"])]
    config["exp_str"] = "_".join(pieces)
    exp_name = "%s_%s" % (ex.path, config["exp_str"])
    ex.observers.append(
        FileStorageObserver.create(
            os.path.join(config["path"]["log_dir"], exp_name)
        )
    )
    return config
