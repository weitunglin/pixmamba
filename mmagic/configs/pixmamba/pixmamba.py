optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.5)))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=800, convert_to_iter_based=True),]

auto_scale_lr=dict(_delete_=True,base_batch_size=8, enable=False)

train_cfg = dict(by_epoch=True, max_epochs=800)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='MultiTestLoop')

find_unused_parameters = True
