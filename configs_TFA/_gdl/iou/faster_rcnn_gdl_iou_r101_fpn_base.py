_base_ = [
    './faster_rcnn_gdl_iou_r50_fpn_base.py',
    '../_base_/data/TFA_voc/split1_base.py',
    '../_base_/schedules/TFA_voc/schedule_1x_base.py', '../_base_/default_runtime.py'
]
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))