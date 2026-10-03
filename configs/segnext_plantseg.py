"""SegNeXt + PlantSeg config template for the current PlantSeg-style image/PNG-mask layout.
Expected layout after extraction:

<PLANTSEG_ROOT>/images/train/*.jpg
<PLANTSEG_ROOT>/images/val/*.jpg
<PLANTSEG_ROOT>/images/test/*.jpg
<PLANTSEG_ROOT>/annotations/train/*.png
<PLANTSEG_ROOT>/annotations/val/*.png
<PLANTSEG_ROOT>/annotations/test/*.png

If your release differs, edit DATA_ROOT/IMG_DIR/ANN_DIR below after running scripts/10_prepare_plantseg.py.
"""
import os

_base_ = [
    '../external/SegNeXt/local_configs/segnext/large/segnext.large.512x512.ade.160k.py'
]

DATA_ROOT = os.environ.get('PLANTSEG_ROOT', 'data/raw/plantseg')
PRETRAINED = os.environ.get('PLANTSEG_MSCAN_L', '')

# PlantSeg disease segmentation: background + disease.
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=None,
)
if PRETRAINED:
    model['backbone'] = dict(init_cfg=dict(type='Pretrained', checkpoint=PRETRAINED))
else:
    model['backbone'] = dict(init_cfg=None)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(512, 512), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root=DATA_ROOT,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        classes=('background','disease'),
        palette=[[0,0,0],[255,255,255]],
    ),
    val=dict(
        type='CustomDataset',
        data_root=DATA_ROOT,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        classes=('background','disease'),
        palette=[[0,0,0],[255,255,255]],
    ),
    test=dict(
        type='CustomDataset',
        data_root=DATA_ROOT,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        classes=('background','disease'),
        palette=[[0,0,0],[255,255,255]],
    ),
)

evaluation = dict(interval=1, metric='mIoU')
checkpoint_config = dict(by_epoch=True, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=80)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9,0.999), weight_decay=0.01)
lr_config = dict(_delete_=True, policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
