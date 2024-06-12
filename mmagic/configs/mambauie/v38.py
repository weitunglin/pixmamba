"""
defaultdict(<class 'float'>, {'conv': 0.150994944, 'layer_norm': 0.23986176, 'upsample_bilinear2d': 0.001572864, 'linear': 3.553175808, 'einsum': 0.72971, 'PythonOp.SelectiveScanFn': 1.094788512})
params 1023346 GFLOPs 5.770103888
"""

_base_ = [
    './seamamba.py'
]

ver = 'v38'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*4,
        dims=[96]*4,
        d_state=16,
        biattn=True,
        biattn_act_ratio=0.125,
        residual=False,
        last_skip=True,
        pixel=False,
        pos_embed=True,
        conv=False,
        p_conv=False,
        p_pixel=True,
        ver='v16',
    ),
    pixel_loss=dict(type='CharbonnierLoss'))

batch_size = 8
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=15, T_max=800, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
