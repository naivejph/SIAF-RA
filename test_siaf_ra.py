import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_siaf_ra_test import ex
from dataloaders.dataset_specifics import get_label_names
from dataloaders.isic import DatasetISIC
from dataloaders.lung import DatasetLung
from models.siaf_ra import FewShotSeg

os.environ["TORCH_HOME"] = "./pretrained_model"

_DATASET_NAME_MAP = {
    "CHAOST2": "ABDOMEN_MR",
    "SABS": "ABDOMEN_CT",
    "CARDIAC_bssFP": "CARDIAC_bssFP",
    "CARDIAC_LGE": "CARDIAC_LGE",
    "MI-PRO": "MI-PRO",
    "Prostate_NCI": "Prostate_NCI",
    "Prostate_UCLH": "Prostate_UCLH",
}

_EVAL_NORMALIZED_SUBDIRS = {
    "Prostate_NCI": "NCI_normalized",
    "Prostate_UCLH": "UCLH_normalized",
}

_2D_DOMAINS = {"ISIC", "Lung"}
_DEFAULT_TARGET_SIZE = 257


def _adnet_dataset_and_dir(config_dataset, config_data_dir):
    adnet_name = _DATASET_NAME_MAP.get(config_dataset, config_dataset)
    data_dir = os.path.normpath(config_data_dir)
    normalized_subdir = _EVAL_NORMALIZED_SUBDIRS.get(adnet_name)
    if normalized_subdir is not None:
        direct_pattern = os.path.join(data_dir, "image*.nii.gz")
        nested_pattern = os.path.join(data_dir, normalized_subdir, "image*.nii.gz")
        if glob.glob(direct_pattern):
            image_pattern = direct_pattern
        elif glob.glob(nested_pattern):
            image_pattern = nested_pattern
        else:
            image_pattern = direct_pattern
    elif adnet_name in {
        "CARDIAC_bssFP", "CARDIAC_LGE", "MI-PRO", "ABDOMEN_MR", "ABDOMEN_CT"
    }:
        image_pattern = os.path.join(data_dir, "image*.nii.gz")
    else:
        image_pattern = os.path.join(data_dir, "*", "image*.nii.gz")
    return adnet_name, image_pattern


def _scan_sort_key(path):
    filename = os.path.basename(path)
    scan_id = filename.replace("image_", "").replace(".nii.gz", "")
    try:
        return 0, int(scan_id)
    except ValueError:
        return 1, scan_id


def resize_volume(volume, target, is_label=False):
    if volume.ndim != 3:
        raise ValueError("Expected [D,H,W], got %s" % (volume.shape,))
    target = int(target)
    if volume.shape[1:] == (target, target):
        return volume
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(1)
    if is_label:
        tensor = F.interpolate(tensor, size=(target, target), mode="nearest")
    else:
        tensor = F.interpolate(tensor, size=(target, target), mode="bilinear", align_corners=False)
    return tensor[:, 0].cpu().numpy()


def dice_score_2d(pred_mask, gt_mask):
    pred_bool = np.asarray(pred_mask).astype(bool)
    gt_bool = np.asarray(gt_mask).astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    denominator = pred_bool.sum() + gt_bool.sum()
    return 1.0 if denominator == 0 else float(2.0 * intersection / denominator)


def _flip_nested_episode(tensors):
    return [[torch.flip(tensor, dims=(-1,)) for tensor in way] for way in tensors]


@torch.no_grad()
def predict_episode(model, support_images, support_fg_mask, support_bg_mask, query_images, use_flip=False):
    logits, _, _ = model(
        support_images, support_fg_mask, support_bg_mask, query_images,
        isval=True, val_wsize=2,
    )
    if not use_flip:
        return logits
    flipped_support = _flip_nested_episode(support_images)
    flipped_fg = _flip_nested_episode(support_fg_mask)
    flipped_bg = _flip_nested_episode(support_bg_mask)
    flipped_query = [torch.flip(tensor, dims=(-1,)) for tensor in query_images]
    flipped_logits, _, _ = model(
        flipped_support, flipped_fg, flipped_bg, flipped_query,
        isval=True, val_wsize=2,
    )
    return 0.5 * (logits + torch.flip(flipped_logits, dims=(-1,)))


@torch.no_grad()
def eval_domain(model, data_dir, device, domain_name, test_label, target_size, use_flip=False):
    adnet_name, image_pattern = _adnet_dataset_and_dir(domain_name, data_dir)
    image_files = sorted(glob.glob(image_pattern), key=_scan_sort_key)
    print("  image pattern : %s" % image_pattern)
    print("  scans found   : %d" % len(image_files))
    print("  input size    : %dx%d" % (target_size, target_size))
    if not image_files:
        raise RuntimeError("No image files found for %s" % domain_name)

    label_names = get_label_names(adnet_name)
    class_dice = {
        name: [] for label_value, name in label_names.items()
        if label_value in test_label and name != "BG"
    }

    for scan_index, image_path in enumerate(image_files, start=1):
        label_path = image_path.replace("image_", "label_")
        if not os.path.isfile(label_path):
            continue
        scan_id = os.path.basename(image_path).replace("image_", "").replace(".nii.gz", "")
        image_volume = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
        label_volume = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int32)
        image_volume = resize_volume(image_volume, target_size, is_label=False)
        label_volume = resize_volume(label_volume, target_size, is_label=True).astype(np.int32)
        image_volume = (image_volume - image_volume.mean()) / (image_volume.std() + 1e-8)

        for label_value, label_name in label_names.items():
            if label_name == "BG" or label_value not in test_label:
                continue
            binary_label = (label_volume == label_value).astype(np.uint8)
            foreground_slices = np.where(binary_label.sum(axis=(1, 2)) > 0)[0]
            if len(foreground_slices) < 2:
                continue

            support_z = int(foreground_slices[len(foreground_slices) // 2])
            support_image = torch.from_numpy(image_volume[support_z]).unsqueeze(0).unsqueeze(0).float().to(device)
            support_image = support_image.repeat(1, 3, 1, 1)
            support_mask = torch.from_numpy(binary_label[support_z].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            support_images = [[support_image]]
            support_fg_mask = [[support_mask[0]]]
            support_bg_mask = [[1.0 - support_mask[0]]]

            slice_dices = []
            for query_z in foreground_slices:
                query_z = int(query_z)
                if query_z == support_z:
                    continue
                query_image = torch.from_numpy(image_volume[query_z]).unsqueeze(0).unsqueeze(0).float().to(device)
                query_image = query_image.repeat(1, 3, 1, 1)
                logits = predict_episode(
                    model, support_images, support_fg_mask, support_bg_mask,
                    [query_image], use_flip=use_flip,
                )
                prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                slice_dices.append(dice_score_2d(prediction, binary_label[query_z]))

            if slice_dices:
                scan_dsc = float(np.mean(slice_dices))
                class_dice[label_name].append(scan_dsc)
                print(
                    "    [%3d/%-3d] scan %s %-20s DSC=%.4f slices=%d support_z=%d"
                    % (scan_index, len(image_files), scan_id, label_name, scan_dsc, len(slice_dices), support_z)
                )

    result = {}
    all_scores = []
    print("\n  ---------- Class summary ----------")
    for label_name, scores in class_dice.items():
        if scores:
            mean_value = float(np.mean(scores))
            std_value = float(np.std(scores))
            result[label_name] = mean_value
            all_scores.extend(scores)
            print("  %-20s: mean DSC = %.4f, std = %.4f (%d scans)" % (
                label_name, mean_value, std_value, len(scores)
            ))
        else:
            result[label_name] = float("nan")
            print("  %-20s: no valid scans" % label_name)
    overall = float(np.mean(all_scores)) if all_scores else float("nan")
    overall_std = float(np.std(all_scores)) if all_scores else float("nan")
    overall_n = int(len(all_scores))
    print("  %-20s: mean DSC = %.4f, std = %.4f (N=%d)" % ("Overall", overall, overall_std, overall_n))
    print("[OVERALL] Mean_DSC=%.6f Std=%.6f N=%d" % (overall, overall_std, overall_n))
    return result, overall


@torch.no_grad()
def eval_domain_2d(model, domain_name, data_dir, device, use_flip=False):
    if domain_name == "ISIC":
        dataset = DatasetISIC(datapath=data_dir, split="val")
        category_names = {0: "nevus", 1: "melanoma", 2: "seborrheic_keratosis"}
    elif domain_name == "Lung":
        dataset = DatasetLung(datapath=data_dir, split="val")
        category_names = {0: "lung"}
    else:
        raise ValueError("Unknown 2-D domain: %s" % domain_name)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    class_dice = {name: [] for name in category_names.values()}
    for episode_index, batch in enumerate(loader, start=1):
        support_images_raw = batch["support_imgs"].to(device)
        support_masks_raw = batch["support_masks"].to(device)
        query_image = batch["query_img"].to(device)
        query_mask = batch["query_mask"].to(device)
        class_id = int(batch["class_id"].item())

        if support_images_raw.dim() == 4:
            support_images_raw = support_images_raw.unsqueeze(1)
        if support_masks_raw.dim() == 3:
            support_masks_raw = support_masks_raw.unsqueeze(1)
        elif support_masks_raw.dim() == 5 and support_masks_raw.shape[2] == 1:
            support_masks_raw = support_masks_raw[:, :, 0]

        number_of_shots = support_images_raw.shape[1]
        support_images = [[support_images_raw[:, shot] for shot in range(number_of_shots)]]
        support_fg_mask = [[support_masks_raw[:, shot] for shot in range(number_of_shots)]]
        support_bg_mask = [[1.0 - support_masks_raw[:, shot] for shot in range(number_of_shots)]]
        logits = predict_episode(
            model, support_images, support_fg_mask, support_bg_mask,
            [query_image], use_flip=use_flip,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(bool)
        ground_truth = query_mask[0].cpu().numpy().astype(bool)
        score = dice_score_2d(prediction, ground_truth)
        class_dice[category_names.get(class_id, str(class_id))].append(score)
        if episode_index == 1 or episode_index % 50 == 0:
            print("    episode %4d/%-4d latest DSC=%.4f" % (episode_index, len(loader), score))

    result = {}
    all_scores = []
    print("\n  ---------- Class summary ----------")
    for name, scores in class_dice.items():
        if scores:
            mean_value = float(np.mean(scores))
            result[name] = mean_value
            all_scores.extend(scores)
            print("  %-20s: mean DSC = %.4f, std = %.4f (%d episodes)" % (
                name, mean_value, float(np.std(scores)), len(scores)
            ))
        else:
            result[name] = float("nan")
    overall = float(np.mean(all_scores)) if all_scores else float("nan")
    overall_std = float(np.std(all_scores)) if all_scores else float("nan")
    overall_n = int(len(all_scores))
    print("  %-20s: mean DSC = %.4f, std = %.4f (N=%d)" % ("Overall", overall, overall_std, overall_n))
    print("[OVERALL] Mean_DSC=%.6f Std=%.6f N=%d" % (overall, overall_std, overall_n))
    return result, overall


@ex.automain
def main(_run, _config, _log):
    device = torch.device(
        "cuda:%d" % _config["gpu_id"] if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = os.path.normpath(_config["reload_model_path"])
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)

    print("========== SIAF-RA Evaluation ==========")
    print("device     : %s" % device)
    print("checkpoint : %s" % checkpoint_path)
    print("flip TTA   : %s" % bool(_config.get("use_horizontal_flip_tta", False)))

    model = FewShotSeg(pretrained_path=checkpoint_path, cfg=_config["model"]).to(device)
    model.eval()
    summary = {}
    for domain_name in _config["eval_domains"]:
        domain_config = _config["path"][domain_name]
        data_dir = domain_config["data_dir"]
        print("\n============================================")
        print("Evaluating domain : %s" % domain_name)
        print("Data directory    : %s" % data_dir)
        if not data_dir or not os.path.exists(data_dir):
            print("[warn] data_dir not found; domain skipped.")
            continue

        use_flip = bool(_config.get("use_horizontal_flip_tta", False))
        if domain_name in _2D_DOMAINS:
            _, mean_dsc = eval_domain_2d(model, domain_name, data_dir, device, use_flip=use_flip)
        else:
            _, mean_dsc = eval_domain(
                model=model,
                data_dir=data_dir,
                device=device,
                domain_name=domain_name,
                test_label=domain_config["test_label"],
                target_size=int(domain_config.get("target_size", _DEFAULT_TARGET_SIZE)),
                use_flip=use_flip,
            )
        summary[domain_name] = mean_dsc
        _run.log_scalar("%s/mean_dsc" % domain_name, mean_dsc)

    print("\n========== Final Summary ==========")
    for domain_name, score in summary.items():
        print("%-20s mean DSC = %.4f" % (domain_name, score))
    _log.info("End of SIAF multi-domain validation")
    return summary
