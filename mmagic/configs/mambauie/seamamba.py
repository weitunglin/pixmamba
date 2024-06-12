_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
]

experiment_name = 'seamamba_uieb'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[2, 2, 2, 2]
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.5)))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=800, convert_to_iter_based=True),]

auto_scale_lr=dict(_delete_=True,base_batch_size=8, enable=True)

train_cfg = dict(by_epoch=True, max_epochs=800)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True
resume = True
