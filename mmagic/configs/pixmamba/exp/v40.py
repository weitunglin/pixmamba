"""
params 8.240 M, FLOPs 7.604 G

(iter 32550)
Start evalutaing T90
PSNR:  23.126570392305595
SSIM:  0.9079349487529392
MSE:  0.006720991158408346
UCIQE:  0.6189887477482724
UIQM:  3.0244727035499572
NIQE:  5.542212017975314
URanker: 2.186109745502472
MUSIQ: 52.26793950398763
Start evalutaing C60
UCIQE:  0.5890394415646327
UIQM:  2.863269816853897
NIQE:  6.246480945755139
URanker: 1.5733595593521992
MUSIQ: 47.875686740875246
Start evalutaing UCCS
UCIQE:  0.5674004559138035
UIQM:  3.05952461978902
NIQE:  4.68853431519631
URanker: 1.47771750714009
MUSIQ: 31.365577017466226
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v40'
experiment_name = f'pixmamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*3,
        dims=128,
        pixel_branch=True,
        bi_scan=True,
        final_refine=False,
        merge_attn=True,
        pos_embed=True,
        last_skip=False,
        patch_size=4,
        mamba_up=True,
        unet_down=False,
        unet_up=False,
        conv_down=False,
        no_act_branch=True,
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
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=15),
    # dict(
    #     type='LinearLR', start_factor=.7, end_factor=.3, by_epoch=True, begin=15, end=16),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=15, end=max_epochs, T_max=max_epochs, convert_to_iter_based=True)]
    # dict(type='LinearLR', by_epoch=True, start_factor=0.5, end_factor=1e-2, begin=max_epochs//2, end=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]

find_unused_parameter=False

# Test Scripts
visualizer = dict(
    type='ConcatImageVisualizer',
    fn_key='img_path',
    img_keys=['pred_img'],
    bgr2rgb=True)


custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1)]
