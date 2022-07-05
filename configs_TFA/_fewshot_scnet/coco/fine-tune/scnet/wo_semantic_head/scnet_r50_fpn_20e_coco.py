_base_ = './scnet_r50_fpn_1x_coco_wo_semantic_head.py'
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
