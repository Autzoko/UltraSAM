_base_ = ['../../_base_/datasets/sam_dataset_bbox_prompt.py',
          '../../_base_/models/sam_mask_refinement.py']

data_root = 'ABUS_2d'

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/train.coco.json',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/val/'),
        ann_file='annotations/val.coco.json',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/test/'),
        ann_file='annotations/test.coco.json',
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    ann_file='{}/annotations/val.coco.json'.format(data_root),
)

test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    ann_file='{}/annotations/test.coco.json'.format(data_root),
)
