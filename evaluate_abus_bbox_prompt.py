"""
Evaluate UltraSAM on ABUS 2D slices with oracle bounding box prompts.

Computes per-slice Dice coefficient and IoU, with per-volume and per-split
aggregation. Uses the MMDet pipeline for data loading and model inference.

Usage:
    python evaluate_abus_bbox_prompt.py \
        --config configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine_ABUS.py \
        --checkpoint UltraSam.pth \
        --split test \
        --data_root ABUS_2d
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import torch
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate UltraSAM on ABUS with oracle bbox prompts')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Which data split to evaluate')
    parser.add_argument('--data_root', type=str, default='ABUS_2d',
                        help='Root directory of converted ABUS 2D data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-slice results CSV')
    return parser.parse_args()


def compute_dice(pred, gt):
    """Compute Dice coefficient between two binary masks."""
    intersection = (pred & gt).sum()
    denominator = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return 2.0 * intersection / denominator


def compute_iou(pred, gt):
    """Compute IoU between two binary masks."""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return intersection / union


def extract_case_id(filename):
    """Extract case_id from filename like DATA_042_slice_0123.png."""
    match = re.match(r'DATA_(\d+)_slice_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    args = parse_args()

    # Load config and override split-specific paths
    cfg = Config.fromfile(args.config)

    split_cfg = {
        'train': dict(
            data_prefix=dict(img='images/train/'),
            ann_file='annotations/train.coco.json',
        ),
        'val': dict(
            data_prefix=dict(img='images/val/'),
            ann_file='annotations/val.coco.json',
        ),
        'test': dict(
            data_prefix=dict(img='images/test/'),
            ann_file='annotations/test.coco.json',
        ),
    }

    # Override test_dataloader for the selected split
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.update(split_cfg[args.split])

    # Build model
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    # Build dataset
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pseudo_collate,
        num_workers=2,
        persistent_workers=True,
    )

    print(f"Evaluating split: {args.split}")
    print(f"Dataset size: {len(dataset)} slices")
    print(f"Device: {args.device}")
    print("-" * 60)

    # Evaluation loop
    slice_results = []
    volume_results = defaultdict(list)

    for i, data_batch in enumerate(dataloader):
        with torch.no_grad():
            results = model.test_step(data_batch)

        for result in results:
            # Get filename for volume tracking
            img_path = result.img_path
            filename = os.path.basename(img_path)
            case_id, slice_idx = extract_case_id(filename)

            # Extract GT mask
            gt_masks = result.gt_instances.masks
            if hasattr(gt_masks, 'masks'):
                gt_mask = gt_masks.masks[0].astype(bool)
            elif isinstance(gt_masks, torch.Tensor):
                gt_mask = gt_masks[0].cpu().numpy().astype(bool)
            else:
                gt_mask = np.array(gt_masks[0]).astype(bool)

            # Extract predicted mask
            if len(result.pred_instances.masks) == 0:
                pred_mask = np.zeros_like(gt_mask)
            else:
                pred_mask = result.pred_instances.masks[0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(bool)

            # Ensure same shape
            if pred_mask.shape != gt_mask.shape:
                from cv2 import resize
                pred_mask = resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=0  # INTER_NEAREST
                ).astype(bool)

            dice = compute_dice(pred_mask, gt_mask)
            iou = compute_iou(pred_mask, gt_mask)

            slice_results.append({
                'filename': filename,
                'case_id': case_id,
                'slice_idx': slice_idx,
                'dice': dice,
                'iou': iou,
                'gt_area': int(gt_mask.sum()),
                'pred_area': int(pred_mask.sum()),
            })

            if case_id is not None:
                volume_results[case_id].append({'dice': dice, 'iou': iou})

            if (i + 1) % 50 == 0:
                running_dice = np.mean([r['dice'] for r in slice_results])
                running_iou = np.mean([r['iou'] for r in slice_results])
                print(f"  [{i + 1}/{len(dataloader)}] "
                      f"Running Dice: {running_dice:.4f}, IoU: {running_iou:.4f}")

    # Aggregate results
    all_dice = [r['dice'] for r in slice_results]
    all_iou = [r['iou'] for r in slice_results]

    print("\n" + "=" * 60)
    print(f"Results for split: {args.split}")
    print(f"  Total slices: {len(slice_results)}")
    print(f"  Mean Dice:  {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"  Mean IoU:   {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"  Median Dice: {np.median(all_dice):.4f}")
    print(f"  Median IoU:  {np.median(all_iou):.4f}")
    print("=" * 60)

    # Per-volume results
    print(f"\nPer-volume results ({len(volume_results)} volumes):")
    print(f"{'Case ID':>10}  {'Slices':>6}  {'Mean Dice':>10}  {'Mean IoU':>10}")
    print("-" * 45)

    vol_dice_means = []
    vol_iou_means = []
    for case_id in sorted(volume_results.keys()):
        metrics = volume_results[case_id]
        vol_dice = np.mean([m['dice'] for m in metrics])
        vol_iou = np.mean([m['iou'] for m in metrics])
        vol_dice_means.append(vol_dice)
        vol_iou_means.append(vol_iou)
        print(f"{case_id:>10}  {len(metrics):>6}  {vol_dice:>10.4f}  {vol_iou:>10.4f}")

    print("-" * 45)
    print(f"{'Volume-avg':>10}  {'':>6}  {np.mean(vol_dice_means):>10.4f}  "
          f"{np.mean(vol_iou_means):>10.4f}")

    # Save per-slice results to CSV
    if args.output_csv is None:
        args.output_csv = f'abus_bbox_eval_{args.split}.csv'

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'case_id', 'slice_idx', 'dice', 'iou', 'gt_area', 'pred_area'])
        writer.writeheader()
        writer.writerows(slice_results)
    print(f"\nPer-slice results saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
