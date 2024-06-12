"""
train 384x384 val 256x256
pixel no bi_scan, 3 to 2 layer
params 8.240 M, FLOPs 7.606 G

(iter 37000)
Start evalutaing T90
PSNR:  22.917082226573854
SSIM:  0.8964149889669382
MSE:  0.0071145259936429926
UCIQE:  0.6201581891580803
UIQM:  3.0594308717914362
NIQE:  5.269395442021099
URanker: 2.2222888520401387
MUSIQ: 51.80966178046332
Start evalutaing C60
UCIQE:  0.5905598604966704
UIQM:  2.8831468509631404
NIQE:  6.514553597905564
URanker: 1.468757801502943
MUSIQ: 47.384653663635255
Start evalutaing UCCS
UCIQE:  0.5661704431658542
UIQM:  3.0895617815684395
NIQE:  4.873290761024995
URanker: 1.38410749649629
MUSIQ: 30.97904507001241
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v52'
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
        pixel_block_num=2,
        pixel_bi_scan=False,
        pixel_d_state=16,
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
        optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 400
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
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [dict(type='BasicVisualizationHook', interval=11)]

find_unused_parameter=False

# Test Scripts
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     fn_key='img_path',
#     img_keys=['pred_img'],
#     bgr2rgb=True)


# custom_hooks = [
#     dict(type='BasicVisualizationHook', interval=1)]
