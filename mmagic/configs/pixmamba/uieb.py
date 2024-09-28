train_img_scale = (256, 256)
val_img_scale = (256, 256)

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
        scale=train_img_scale,
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
        scale=train_img_scale,
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
        scale=val_img_scale,
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
        scale=val_img_scale,
    ),
    dict(type='PackInputs')
]

data_root = '../data/uieb/'

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

t90_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='t90', task_name='denoising'),
        data_root=data_root+'valid_t90',
        data_prefix=dict(img='.', gt='.'),
        pipeline=val_pipeline
    ),
)

c60_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='c60', task_name='denoising'),
        data_root=data_root+'test',
        data_prefix=dict(img='.', gt='.'),
        pipeline=val_pipeline
    ),
)

uccs_blue_dataroot = '../data/uccs/blue'

uccs_blue_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='uccsblue', task_name='denoising'),
        data_root=uccs_blue_dataroot,
        data_prefix=dict(img='.', gt='.'),
        pipeline=val_pipeline
    ),
)

uccs_green_dataroot = '../data/uccs/green'

uccs_green_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='uccsgreen', task_name='denoising'),
        data_root=uccs_green_dataroot,
        data_prefix=dict(img='.', gt='.'),
        pipeline=val_pipeline
    ),
)


uccs_blue_green_dataroot = '../data/uccs/blue-green'

uccs_blue_green_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='uccsbluegreen', task_name='denoising'),
        data_root=uccs_blue_green_dataroot,
        data_prefix=dict(img='.', gt='.'),
        pipeline=val_pipeline
    ),
)


evaluator = [
    dict(type='MAE', prefix='uie'),
    dict(type='MSE', prefix='uie'),
    dict(type='SSIM', prefix='uie'),
    dict(type='PSNR', prefix='uie'),
]


c60_evaluator = [
    dict(type='MAE', prefix='60'),
    # dict(type='UIQM', prefix='c60'),
    # dict(type='UCIQE', prefix='c60'),
]
t90_evaluator = [
    dict(type='MAE', prefix='t90'),
]
uccsgreen_evaluator = [
    dict(type='MAE', prefix='uccsgreen'),
]
uccsblue_evaluator = [
    dict(type='MAE', prefix='uccsblue'),
]
uccsbluegreen_evaluator = [
    dict(type='MAE', prefix='uccsbluegreen'),
]

val_evaluator = evaluator

test_dataloader = [t90_dataloader, c60_dataloader, uccs_green_dataloader, uccs_blue_dataloader, uccs_blue_green_dataloader]
test_evaluator = [t90_evaluator,c60_evaluator,uccsgreen_evaluator,uccsblue_evaluator,uccsbluegreen_evaluator]

