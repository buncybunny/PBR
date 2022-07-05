_base_ = './faster_rcnn_r50_fpn_1x_coco_tfa.py'
model = dict(train_cfg=dict(rcnn=dict(sampler=dict(type='OHEMSampler'))))
