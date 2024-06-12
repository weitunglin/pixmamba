"""
defaultdict(<class 'float'>, {'conv': 0.4230144, 'layer_norm': 0.62208, 'linear': 3.906693504, 'einsum': 1.29381, 'PythonOp.SelectiveScanFn': 1.8413208})
params 117438 GFLOPs 8.086918703999999
"""

_base_ = [
    './seamamba.py'
]

ver = 'v20'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*6,
        dims=[36]*6,
        d_state=10,
        biattn_act_ratio=0.25,
        ver='v16',
    ),
    pixel_loss=dict(type='CharbonnierLoss'))

batch_size = 32
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.00025, betas=(0.9, 0.999))))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=15),
    dict(
        type='LinearLR', start_factor=1, end_factor=0.5, by_epoch=True, begin=15, end=max_epochs//2),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=max_epochs//2, T_max=max_epochs, convert_to_iter_based=True),]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))

custom_hooks = [dict(type='BasicVisualizationHook', interval=4)]
