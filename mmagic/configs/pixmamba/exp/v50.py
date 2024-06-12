"""
params 8.241 M, FLOPs 7.643 G

(iter 36100)
Start evalutaing T90
PSNR:  23.39666420466293
SSIM:  0.9099078549225632
MSE:  0.006403655371841881
UCIQE:  0.6183308587956616
UIQM:  3.0207655095746544
NIQE:  5.574165919214101
URanker: 2.2118712766302955
MUSIQ: 52.47874018351237
Start evalutaing C60
UCIQE:  0.5912655624275042
UIQM:  2.7884277165345996
NIQE:  6.555670563797087
URanker: 1.5196840934455396
MUSIQ: 48.068664360046384
Start evalutaing UCCS
UCIQE:  0.5636819393474521
UIQM:  3.007517429538927
NIQE:  4.760023414084809
URanker: 1.3739629413746297
MUSIQ: 31.172834075291952
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v50'
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
        pixel_block_num=3,
        pixel_bi_scan=True,
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
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean', eps=1e-10),
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
        optimizer=dict(type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=20),
    # dict(
    #     type='LinearLR', start_factor=.7, end_factor=.3, by_epoch=True, begin=15, end=16),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=20, end=max_epochs, T_max=max_epochs, convert_to_iter_based=True)]
    # dict(type='LinearLR', by_epoch=True, start_factor=0.5, end_factor=1e-2, begin=max_epochs//2, end=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
custom_hooks = [dict(type='BasicVisualizationHook', interval=10)]

find_unused_parameter=False

# Test Scripts
visualizer = dict(
    type='ConcatImageVisualizer',
    fn_key='img_path',
    img_keys=['pred_img'],
    bgr2rgb=True)


custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1)]
