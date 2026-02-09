"""
Convert 3D ABUS NRRD volumes to 2D slices in COCO annotation format.

For each volume, slices along the elevation axis (axis 2) are extracted.
Only slices containing tumor (mask > 0) are saved. The bounding box is
computed as the minimum enclosing axis-aligned rectangle of the mask.

Usage:
    python convert_abus_to_2d.py \
        --input_dir /Volumes/Autzoko/Dataset/ABUS/data \
        --output_dir ABUS_2d
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import cv2
import nrrd
import numpy as np
from pycocotools import mask as maskUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SPLIT_DIRS = {
    'train': 'Train',
    'val': 'Validation',
    'test': 'Test',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert 3D ABUS NRRD to 2D COCO format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory of ABUS data (containing Train/Validation/Test)')
    parser.add_argument('--output_dir', type=str, default='ABUS_2d',
                        help='Output directory for converted data')
    parser.add_argument('--slice_axis', type=int, default=2,
                        help='Axis along which to slice 3D volumes (default: 2, elevation)')
    return parser.parse_args()


def read_labels_csv(csv_path):
    """Read labels.csv and return list of (case_id, label, data_path, mask_path)."""
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'case_id': int(row['case_id']),
                'label': row['label'],
                'data_path': row['data_path'].replace('\\', '/'),
                'mask_path': row['mask_path'].replace('\\', '/'),
            })
    return entries


def compute_bbox_from_mask(mask_2d):
    """Compute tight axis-aligned bounding box from a binary 2D mask.

    Returns [x, y, w, h] in COCO format (x=col_min, y=row_min).
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def encode_mask_rle(mask_2d):
    """Encode a binary 2D mask as COCO RLE."""
    mask_fortran = np.asfortranarray(mask_2d.astype(np.uint8))
    rle = maskUtils.encode(mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def extract_slice(volume, slice_idx, axis):
    """Extract a 2D slice from a 3D volume along the given axis."""
    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    return volume[tuple(slicing)]


def convert_split(input_dir, output_dir, split_name, split_dir_name, slice_axis):
    """Convert one split of ABUS data to 2D COCO format."""
    split_input = Path(input_dir) / split_dir_name
    csv_path = split_input / 'labels.csv'
    entries = read_labels_csv(csv_path)

    img_out_dir = Path(output_dir) / 'images' / split_name
    os.makedirs(img_out_dir, exist_ok=True)
    ann_out_dir = Path(output_dir) / 'annotations'
    os.makedirs(ann_out_dir, exist_ok=True)

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for entry in entries:
        case_id = entry['case_id']
        data_path = split_input / entry['data_path']
        mask_path = split_input / entry['mask_path']

        logging.info(f"[{split_name}] Processing case {case_id}: {data_path.name}")

        data, _ = nrrd.read(str(data_path))
        mask, _ = nrrd.read(str(mask_path))

        n_slices = data.shape[slice_axis]
        slices_with_tumor = 0

        for z in range(n_slices):
            mask_slice = extract_slice(mask, z, slice_axis)
            if not mask_slice.any():
                continue

            img_slice = extract_slice(data, z, slice_axis)
            height, width = img_slice.shape[:2]

            # Save image as PNG
            filename = f"DATA_{case_id:03d}_slice_{z:04d}.png"
            cv2.imwrite(str(img_out_dir / filename), img_slice)

            # Compute bbox from mask
            bbox = compute_bbox_from_mask(mask_slice)
            area = float(mask_slice.sum())

            # Encode mask as RLE
            rle = encode_mask_rle(mask_slice)

            # Add image entry
            images.append({
                'id': image_id,
                'file_name': filename,
                'height': height,
                'width': width,
            })

            # Add annotation entry
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox,
                'area': area,
                'segmentation': rle,
                'iscrowd': 0,
            })

            image_id += 1
            annotation_id += 1
            slices_with_tumor += 1

        logging.info(f"  -> {slices_with_tumor} slices with tumor (out of {n_slices})")

    # Build COCO JSON
    coco_dict = {
        'info': {
            'description': f'ABUS 2D slices - {split_name} split',
            'url': '',
        },
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'tumor', 'supercategory': 'lesion'}],
    }

    json_path = ann_out_dir / f'{split_name}.coco.json'
    with open(json_path, 'w') as f:
        json.dump(coco_dict, f)

    logging.info(f"[{split_name}] Saved {len(images)} images, {len(annotations)} annotations -> {json_path}")
    return len(images), len(annotations)


def main():
    args = parse_args()

    total_images = 0
    total_annotations = 0

    for split_name, split_dir_name in SPLIT_DIRS.items():
        n_img, n_ann = convert_split(
            args.input_dir, args.output_dir, split_name, split_dir_name, args.slice_axis)
        total_images += n_img
        total_annotations += n_ann

    logging.info(f"Done. Total: {total_images} images, {total_annotations} annotations across all splits.")


if __name__ == '__main__':
    main()
