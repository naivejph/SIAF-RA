import math
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config_siaf_ra_train import ex
from dataloaders.datasets import TrainDataset
from models.siaf_ra import FewShotSeg
from util.utils import compose_wt_simple, set_seed

os.environ["TORCH_HOME"] = "./pretrained_model"

_DATASET_NAME_MAP = {
    "CHAOST2": "ABDOMEN_MR",
    "SABS": "ABDOMEN_CT",
    "CARDIAC_bssFP": "CARDIAC_bssFP",
    "CARDIAC_LGE": "CARDIAC_LGE",
    "Prostate_NCI": "Prostate_NCI",
    "Prostate_UCLH": "Prostate_UCLH",
}

_TRAIN_NORMALIZED_SUBDIRS = {
    "ABDOMEN_MR": "chaos_MR_T2_normalized",
    "ABDOMEN_CT": "sabs_CT_normalized",
    "CARDIAC_bssFP": "cmr_bssFP_normalized",
    "CARDIAC_LGE": "cmr_LGE_normalized",
    "Prostate_NCI": "NCI_normalized",
    "Prostate_UCLH": "UCLH_normalized",
}


def _adnet_dataset_and_dir(config_dataset, config_data_dir):
    adnet_name = _DATASET_NAME_MAP.get(config_dataset, config_dataset)
    data_dir = os.path.normpath(config_data_dir)
    if adnet_name not in _TRAIN_NORMALIZED_SUBDIRS:
        raise ValueError("Unsupported training dataset: %s" % config_dataset)
    normalized_name = _TRAIN_NORMALIZED_SUBDIRS[adnet_name]
    if os.path.basename(data_dir) == normalized_name:
        data_root = os.path.dirname(data_dir)
    elif os.path.isdir(os.path.join(data_dir, normalized_name)):
        data_root = data_dir
    else:
        data_root = data_dir
    return adnet_name, data_root


def soft_dice_loss(logits, target, ignore_index=255, eps=1e-6):
    probability = torch.softmax(logits, dim=1)[:, 1]
    valid = target != ignore_index
    foreground = (target == 1).float()
    probability = probability * valid.float()
    foreground = foreground * valid.float()
    intersection = (probability * foreground).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + foreground.sum(dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def freeze_backbone_batchnorm(model):
    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_optimizer(model, optim_cfg):
    base_lr = float(optim_cfg.get("lr", 1e-3))
    backbone_mult = float(optim_cfg.get("backbone_lr_mult", 0.10))
    momentum = float(optim_cfg.get("momentum", 0.9))
    weight_decay = float(optim_cfg.get("weight_decay", 5e-4))

    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone if name.startswith("encoder.") else head).append(param)

    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "lr": base_lr * backbone_mult,
            "weight_decay": weight_decay,
            "group_name": "backbone",
        })
    if head:
        groups.append({
            "params": head,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "siaf_ra",
        })

    print("###### Optimizer groups ######")
    for group in groups:
        print("%-12s lr=%.2e params=%d" % (
            group["group_name"], group["lr"],
            sum(p.numel() for p in group["params"]),
        ))
    return torch.optim.SGD(groups, momentum=momentum)


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        snapshot_dir = "%s/snapshots" % _run.observers[0].dir
        os.makedirs(snapshot_dir, exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            destination = os.path.join(_run.observers[0].dir, "source", source_file)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            _run.observers[0].save_file(source_file, os.path.join("source", source_file))
        source_cache = os.path.join(_run.observers[0].basedir, "_sources")
        if os.path.exists(source_cache):
            shutil.rmtree(source_cache)

    set_seed(_config["seed"])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config["gpu_id"])
    torch.set_num_threads(1)

    _log.info("###### Create SIAF-RA ######")
    model = FewShotSeg(cfg=_config["model"]).cuda()
    model.train()
    if bool(_config.get("freeze_backbone_bn", True)):
        freeze_backbone_batchnorm(model)

    adnet_dataset, adnet_dir = _adnet_dataset_and_dir(
        _config["dataset"],
        _config["path"][_config["dataset"]]["data_dir"],
    )
    data_config = {
        "data_dir": adnet_dir,
        "dataset": adnet_dataset,
        "n_shot": _config["task"]["n_shots"],
        "n_way": _config["task"]["n_ways"],
        "n_query": _config["task"]["n_queries"],
        "n_sv": _config["n_sv"],
        "max_iter": _config["max_iters_per_load"],
        "min_size": _config["min_size"],
        "test_label": _config["test_label"],
        "exclude_label": _config["exclude_label"],
        "use_gt": _config["use_gt"],
    }
    train_dataset = TrainDataset(data_config)
    trainloader = DataLoader(
        train_dataset,
        batch_size=_config["batch_size"],
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    optimizer = build_optimizer(model, _config["optim"])
    scheduler = MultiStepLR(
        optimizer,
        milestones=_config["lr_milestones"],
        gamma=_config["lr_step_gamma"],
    )

    if _config["dataset"] in ("Prostate_NCI", "Prostate_UCLH"):
        class_weight = (
            torch.FloatTensor([0.05, 1.0]).cuda()
            if _config["use_wce"] else torch.ones(2).cuda()
        )
    else:
        class_weight = compose_wt_simple(_config["use_wce"], _config["dataset"])
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config["ignore_label"], weight=class_weight
    )

    iteration = 0
    num_sub_epochs = _config["n_steps"] // _config["max_iters_per_load"] + 1
    log_loss = {"final_ce": 0.0, "final_dice": 0.0, "sem_ce": 0.0, "sem_dice": 0.0, "total": 0.0}
    _log.info("###### Training SIAF-RA ######")

    for sub_epoch in range(num_sub_epochs):
        _log.info("###### Epoch %d/%d ######" % (sub_epoch, num_sub_epochs))
        for sample_batched in trainloader:
            iteration += 1

            support_image_list = sample_batched["support_images"][0]
            support_mask_list = sample_batched["support_fg_labels"][0]
            support_images_raw = torch.cat(
                [image[:, 0:1, :, :] for image in support_image_list], dim=0
            ).unsqueeze(0).float().cuda(non_blocking=True)
            support_masks_raw = torch.cat(
                list(support_mask_list), dim=0
            ).unsqueeze(0).float().cuda(non_blocking=True)
            query_image_raw = sample_batched["query_images"][0][:, 0:1].float().cuda(non_blocking=True)
            query_labels = sample_batched["query_labels"][0].long().cuda(non_blocking=True)
            if query_labels.dim() == 2:
                query_labels = query_labels.unsqueeze(0)

            _, num_shots, _, _, _ = support_images_raw.shape
            support_images_3ch = support_images_raw.repeat(1, 1, 3, 1, 1)
            query_image_3ch = query_image_raw.repeat(1, 3, 1, 1)
            support_images = [[support_images_3ch[0, s].unsqueeze(0) for s in range(num_shots)]]
            support_fg_mask = [[support_masks_raw[0, s].unsqueeze(0) for s in range(num_shots)]]
            support_bg_mask = [[(1.0 - support_masks_raw[0, s]).unsqueeze(0) for s in range(num_shots)]]

            optimizer.zero_grad(set_to_none=True)
            try:
                prediction, semantic_prediction, _ = model(
                    support_images,
                    support_fg_mask,
                    support_bg_mask,
                    [query_image_3ch],
                    isval=False,
                    query_gt=query_labels,
                )
            except Exception as error:
                print("Faulty batch, skipped: %s: %s" % (type(error).__name__, error))
                optimizer.zero_grad(set_to_none=True)
                continue

            final_ce = criterion(prediction, query_labels)
            final_dice = soft_dice_loss(prediction, query_labels, _config["ignore_label"])
            sem_ce = criterion(semantic_prediction, query_labels)
            sem_dice = soft_dice_loss(semantic_prediction, query_labels, _config["ignore_label"])

            semantic_anchor = sem_ce + float(_config.get("semantic_anchor_dice", 0.5)) * sem_dice
            total_loss = (
                final_ce
                + float(_config.get("lambda_dice", 0.5)) * final_dice
                + float(_config.get("lambda_semantic_anchor", 0.5)) * semantic_anchor
            )

            if not torch.isfinite(total_loss):
                print("Non-finite loss at step %d; skipped." % iteration)
                continue

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(_config.get("gradient_clip_norm", 5.0))
            )
            if not math.isfinite(float(grad_norm)):
                print("Non-finite gradient at step %d; skipped." % iteration)
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            scheduler.step()
            if bool(_config.get("freeze_backbone_bn", True)):
                freeze_backbone_batchnorm(model)

            values = {
                "final_ce": float(final_ce.detach()),
                "final_dice": float(final_dice.detach()),
                "sem_ce": float(sem_ce.detach()),
                "sem_dice": float(sem_dice.detach()),
                "total": float(total_loss.detach()),
            }
            for key, value in values.items():
                _run.log_scalar(key, value)
                log_loss[key] += value

            _run.log_scalar("semantic_scale", float(model.matcher.semantic_scale().detach()))
            _run.log_scalar("field_scale", float(model.matcher.field_scale().detach()))
            if model._last_siaf_ra:
                _run.log_scalar("mean_trust", float(model._last_siaf_ra["trust"].mean()))
                _run.log_scalar("quality_separation", float(model._last_siaf_ra["quality_separation"].mean()))

            if iteration % _config["print_interval"] == 0:
                n = float(_config["print_interval"])
                head_lr = max(g["lr"] for g in optimizer.param_groups)
                backbone_lr = min(g["lr"] for g in optimizer.param_groups)
                mean_trust = float(model._last_siaf_ra["trust"].mean())
                qsep = float(model._last_siaf_ra["quality_separation"].mean())
                print(
                    "step %d: total=%.4f final_ce=%.4f final_dice=%.4f "
                    "sem_ce=%.4f sem_dice=%.4f lr_head=%.2e lr_backbone=%.2e "
                    "semT=%.3f fieldS=%.3f trust=%.3f qsep=%.4f"
                    % (
                        iteration,
                        log_loss["total"] / n,
                        log_loss["final_ce"] / n,
                        log_loss["final_dice"] / n,
                        log_loss["sem_ce"] / n,
                        log_loss["sem_dice"] / n,
                        head_lr,
                        backbone_lr,
                        float(model.matcher.semantic_scale().detach()),
                        float(model.matcher.field_scale().detach()),
                        mean_trust,
                        qsep,
                    )
                )
                log_loss = {key: 0.0 for key in log_loss}

            if iteration % _config["max_iters_per_load"] == 0:
                if hasattr(trainloader.dataset, "reload_buffer"):
                    trainloader.dataset.reload_buffer()
                    print("###### Dataset reloaded ######")

            # Keep ONLY the final checkpoint.
            if iteration >= _config["n_steps"]:
                snapshot_dir = "%s/snapshots" % _run.observers[0].dir
                final_path = os.path.join(snapshot_dir, "%d.pth" % _config["n_steps"])
                _log.info("###### Taking FINAL snapshot: %s ######" % final_path)
                torch.save(model.state_dict(), final_path)
                for filename in os.listdir(snapshot_dir):
                    if filename.endswith(".pth") and filename != "%d.pth" % _config["n_steps"]:
                        try:
                            os.remove(os.path.join(snapshot_dir, filename))
                        except OSError:
                            pass
                return 1

    return 1
