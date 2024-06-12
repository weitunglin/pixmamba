"""
v26
dim 96
lr * 2
pos_embed true
pixel_pos_embed true
final_refine false
bi_scan true
pixel_bi_scan false
add norm in mamba_up
params 4.755 M, FLOPs 4.712 G

(iter 65100)
Start evalutaing T90
PSNR:  23.02754875151631
SSIM:  0.9036804840834248
MSE:  0.006710717418144353
UCIQE:  0.6134897302942766
UIQM:  3.0641833587405647
NIQE:  5.5666267961583396
URanker: 2.106252442465888
MUSIQ: 52.226990932888455

Start evalutaing C60
UCIQE:  0.5850501401381399
UIQM:  2.847291724901027
NIQE:  6.463686736791108
URanker: 1.4492208523054917
MUSIQ: 47.89262323379516

Start evalutaing UCCS
UCIQE:  0.5632138140141492
UIQM:  3.0786805874585186
NIQE:  4.747039970662977
URanker: 1.4191775837556149
MUSIQ: 31.20110596338908

(iter 67300)
Start evalutaing T90
PSNR:  23.01197671609066
SSIM:  0.9054121777430139
MSE:  0.006689331085106434
UCIQE:  0.616296777129065
UIQM:  3.0215261012149104
NIQE:  5.54379714448092
URanker: 2.1446015222205057
MUSIQ: 52.23958034515381

Start evalutaing C60
UCIQE:  0.5862219578172163
UIQM:  2.8030652651504915
NIQE:  6.430014264353377
URanker: 1.4739552782538037
MUSIQ: 47.81066004435221

Start evalutaing UCCS
UCIQE:  0.5648749016108965
UIQM:  3.053188766680085
NIQE:  4.751180645290585
URanker: 1.471603610276555
MUSIQ: 31.151692145665486
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './pixmamba.py'
]

ver = 'v35'
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

batch_size = 8
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
    dict(type='CosineAnnealingLR', by_epoch=True, begin=15, T_max=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [dict(type='BasicVisualizationHook', interval=15)]

find_unused_parameter=False

# Test Scripts
visualizer = dict(
    type='ConcatImageVisualizer',
    fn_key='img_path',
    img_keys=['pred_img'],
    bgr2rgb=True)


custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1)]
