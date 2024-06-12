"""
defaultdict(<class 'float'>, {'conv': 0.4644864, 'layer_norm': 0.940032, 'linear': 1.636762752, 'einsum': 1.59247, 'PythonOp.SelectiveScanFn': 2.6541248})
params 5679 GFLOPs 7.287875952
"""

_base_ = [
    './seamamba.py'
]

ver = 'v29'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*1,
        dims=[24]*1,
        d_state=6,
        biattn_act_ratio=0.5,
        residual=False,
        last_skip=True,
        pixel=True,
        pos_embed=True,
        ver='v16',
    ),
    pixel_loss=dict(type='CharbonnierLoss'))

batch_size = 32
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
    dict(type='LinearLR', start_factor=1, end_factor=1e-3, by_epoch=True, begin=15, end=800)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
