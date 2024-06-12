img_scale = (256, 256)

pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=img_scale,
    ),
    dict(type='PackInputs')
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=img_scale,
    ),
    dict(type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=img_scale,
    ),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        keys=['img'],
        scale=img_scale,
    ),
    dict(type='PackInputs')
]

data_root = '/home/allen/workspace/seamamba/data/uieb_t90/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='UIEB', task_name='denoising'),
        data_root=data_root+'train',
        data_prefix=dict(img='raw-890', gt='reference-890'),
        pipeline=train_pipeline
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='UIEB', task_name='denoising'),
        data_root=data_root+'valid',
        data_prefix=dict(img='raw-890', gt='reference-890'),
        pipeline=val_pipeline
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='UIEB', task_name='denoising'),
        data_root=data_root+'test',
        data_prefix=dict(img='.'),
        pipeline=test_pipeline
    ),
)

evaluator = [
    dict(type='MAE', prefix='uie'),
    dict(type='MSE', prefix='uie'),
    dict(type='SSIM', prefix='uie'),
    dict(type='PSNR', prefix='uie'),
]

val_evaluator = evaluator

test_evaluator = []
