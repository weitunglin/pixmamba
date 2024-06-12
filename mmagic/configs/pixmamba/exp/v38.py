"""
bi_scan true
pos_embed false
params 4.755 M, FLOPs 4.713 G

(iter 16425)
Start evalutaing T90
PSNR:  22.99023227080664
SSIM:  0.9029651190473411
MSE:  0.006858068074841507
UCIQE:  0.6159108616979333
UIQM:  3.020636503331273
NIQE:  5.645867141907957
URanker: 2.210869132147895
MUSIQ: 52.72229376898871
Start evalutaing C60
UCIQE:  0.5854725446276899
UIQM:  2.804497779544785
NIQE:  6.5719054823492815
URanker: 1.4993660245711604
MUSIQ: 48.780076313018796
Start evalutaing UCCS
UCIQE:  0.5652003247254334
UIQM:  3.0859622618414186
NIQE:  4.639751633715395
URanker: 1.445492791168702
MUSIQ: 31.61474608739217
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v38'
experiment_name = f'pixmamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*3,
        dims=96,
        pixel_branch=True,
        bi_scan=True,
        final_refine=False,
        merge_attn=True,
        pos_embed=False,
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

batch_size = 32
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=15),
    dict(
        type='LinearLR', start_factor=1, end_factor=0.5, by_epoch=True, begin=15, end=16),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=16, end=max_epochs, T_max=max_epochs, convert_to_iter_based=True)]
    # dict(type='LinearLR', by_epoch=True, start_factor=0.5, end_factor=1e-2, begin=max_epochs//2, end=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=3))
custom_hooks = [dict(type='BasicVisualizationHook', interval=3)]

find_unused_parameter=False

# Test Scripts
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     fn_key='img_path',
#     img_keys=['pred_img'],
#     bgr2rgb=True)


# custom_hooks = [
#     dict(type='BasicVisualizationHook', interval=1)]
