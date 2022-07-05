# optimizer
optimizer = dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.00003)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[950,998])
runner = dict(type='EpochBasedRunner', max_epochs=1000)
