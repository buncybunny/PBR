_base_ = './faster_rcnn_r50_fpn_1x_voc_split1_ft_all_cosine_3shot.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
