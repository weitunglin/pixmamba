"""
defaultdict(<class 'float'>, {'conv': 0.088080384, 'layer_norm': 0.13893632, 'upsample_bilinear2d': 0.001048576, 'linear': 0.774931456, 'einsum': 0.452967, 'PythonOp.SelectiveScanFn': 0.729815008})
params 303418 GFLOPs 2.1857787440000003
"""

_base_ = [
    './seamamba.py'
]

ver = 'v39'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*4,
        dims=[64]*4,
        d_state=16,
        bidir='a',
        biattn=True,
        biattn_act_ratio=0.125,
        residual=False,
        last_skip=False,
        pixel=False,
        pos_embed=True,
        conv=False,
        p_conv=False,
        p_pixel=True,
        ver='v16',
    ),
    pixel_loss=dict(type='CharbonnierLoss'))

batch_size = 16
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=15, T_max=800, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]
