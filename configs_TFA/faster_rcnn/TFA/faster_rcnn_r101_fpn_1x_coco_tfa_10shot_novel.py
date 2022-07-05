_base_ = './faster_rcnn_r50_fpn_1x_coco_tfa_10shot_novel.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
