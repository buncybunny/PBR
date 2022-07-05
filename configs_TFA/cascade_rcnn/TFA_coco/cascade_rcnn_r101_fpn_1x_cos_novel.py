_base_ = './cascade_rcnn_r50_fpn_1x_cos_base.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
