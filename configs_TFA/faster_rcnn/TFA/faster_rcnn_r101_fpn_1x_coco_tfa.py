_base_ = './faster_rcnn_r50_fpn_1x_coco_tfa.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
