"""
params 8.241 M, FLOPs 7.643 G

(iter 59800 test on 256x256)
Start evalutaing T90
PSNR:  22.831072553060697
SSIM:  0.9013285658419665
MSE:  0.006889969646666736
UCIQE:  0.6179543246266418
UIQM:  3.0469949950558943
NIQE:  5.4979485123726715
URanker: 2.1305204598440066
MUSIQ: 52.46729007297092
Start evalutaing C60
UCIQE:  0.5895132851745842
UIQM:  2.847159549255491
NIQE:  6.448873750176639
URanker: 1.4232112276057403
MUSIQ: 48.11830498377482
Start evalutaing UCCS
UCIQE:  0.5568863591068991
UIQM:  3.0422087049876168
NIQE:  4.895443960043054
URanker: 1.3012771041815479
MUSIQ: 31.79857967376709
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v51'
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

batch_size = 8
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.00015, betas=(0.9, 0.999), weight_decay=0.5)))

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
custom_hooks = [dict(type='BasicVisualizationHook', interval=20)]

find_unused_parameter=False

# Test Scripts
visualizer = dict(
    type='ConcatImageVisualizer',
    fn_key='img_path',
    img_keys=['pred_img'],
    bgr2rgb=True)


custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1)]
