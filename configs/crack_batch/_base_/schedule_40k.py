# 40k-iteration schedule shared by the crack batch experiments.
# Only the loop/validation/checkpoint settings live here; each algorithm keeps
# its own optimizer from the original MMSeg config.
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Keep only the best checkpoint (by mIoU).  interval is larger than
# max_iters, so no regular iter_*.pth files are written; save_last=False
# disables last.pth.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=40001,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
        save_last=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
