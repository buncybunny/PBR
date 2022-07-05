from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# import matplotlib.pyplot as plt
import cv2
import glob
import os
import sys
import time
import argparse

'''-
python tools/my_inference.py
--cfg configs/my_config/htc_without_semantic_r50_fpn_1x_coco_linux.py
--output-dir work_dirs/htc/20200608/infer_paper/
--wts work_dirs/htc/20200608/epoch_200.pth
../coco/infer_img_dir
'''


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.3,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):

    config_file = args.cfg
    checkpoint_file = args.weights
    thresh = args.thresh
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # model = init_detector(config_file, checkpoint_file, device='cpu')

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    total_t = time.time()
    for i, im_name in enumerate(im_list):
        print('Get img {}'.format(im_name))
        one_t = time.time()
        if (args.image_ext == args.output_ext):
            output_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(im_name))
            )
        else:
            output_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
            )

        result = inference_detector(model, im_name)
        if hasattr(model, 'module'):
            model = model.module
        pred_img = model.show_result(im_name, result, score_thr=thresh, show=False)
        cv2.imwrite(output_name, pred_img)
        print('write {}'.format(output_name))
        print('One image time {:.3f}s'.format(time.time() - one_t))
    print("Total inference time is {:.3f}s".format(time.time() - total_t))


args = parse_args()
main(args)