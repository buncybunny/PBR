_base_ = [
    '../../_base_/models/TFA_coco/cascade_rcnn_r50_fpn_cos_novel.py',
    '../../_base_/data/TFA_coco/3shot_novel.py',
    '../../_base_/schedules/TFA_coco/schedule_1x_3shot_novel.py', '../../_base_/default_runtime.py'
]
