"""
defaultdict(<class 'float'>, {'conv': 0.41472, 'layer_norm': 0.55296, 'linear': 4.329697536, 'einsum': 1.26074, 'PythonOp.Sele
ctiveScanFn': 1.9242784})
params 120100 GFLOPs 8.482395936
"""

_base_ = [
    './seamamba.py'
]

ver = 'v13'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*4,
        dims=[48]*4,
        ver=ver,
    ))

batch_size = 32
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

custom_hooks = [dict(type='BasicVisualizationHook', interval=3)]

optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True),]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=True)