_base_ = [
    '../../_base_/models/TFA_voc/faster_rcnn_r50_fpn_cos_all.py',
    '../../_base_/data/TFA_voc/split1_ft_all.py',
    '../../_base_/schedules/TFA_voc/schedule_1x_split1_ft_all_3shot.py', '../../_base_/default_runtime.py'
]
