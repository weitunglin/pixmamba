"""
no pixmambabranch, no mambaup, no biscan, no mergeattn
params 7.343 M, FLOPs 8.034 G
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'ablate_a'
experiment_name = f'pixmamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*3,
        dims=128,
        pixel_branch=False,
        pixel_block_num=2,
        pixel_bi_scan=False,
        pixel_d_state=12,
        bi_scan=False,
        final_refine=False,
        merge_attn=False,
        pos_embed=True,
        last_skip=True,
        patch_size=4,
        mamba_up=False,
        unet_down=False,
        unet_up=True,
        conv_down=False,
        no_act_branch=False,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    )
)

batch_size = 16
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=20),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=20, end=max_epochs, T_max=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]

find_unused_parameter=False

# Test Scripts
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     fn_key='img_path',
#     img_keys=['pred_img'],
#     bgr2rgb=True)


# custom_hooks = [
#     dict(type='BasicVisualizationHook', interval=1)]
