# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch.nn as nn
from torch.autograd import Function
import torch


class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


class AffineLayer(nn.Module):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, X):
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(X)
        return out


def decouple_layer(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)


@DETECTORS.register_module()
class CascadeRCNN_GDL(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CascadeRCNN_GDL, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.affine_rpn = AffineLayer(num_channels=256, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=256, bias=True)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()
        rpn_scale = 0
        if len(self.CLASSES) in [15, 60]:
            rcnn_scale = 0.75
        elif len(self.CLASSES) in [20, 80]:
            rcnn_scale = 0.001
        features_de_rpn = tuple([self.affine_rpn(decouple_layer(k, rpn_scale)) for k in x])
        features_de_rcnn = tuple([self.affine_rcnn(decouple_layer(k, rcnn_scale)) for k in x])
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                features_de_rpn,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(features_de_rcnn, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        rpn_scale = 0
        if len(self.CLASSES) in [15, 60]:
            rcnn_scale = 0.75
        elif len(self.CLASSES) in [20, 80]:
            rcnn_scale = 0.001
        features_de_rpn = tuple([self.affine_rpn(decouple_layer(k, rpn_scale)) for k in x])
        features_de_rcnn = tuple([self.affine_rcnn(decouple_layer(k, rcnn_scale)) for k in x])
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(features_de_rpn, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            features_de_rcnn, proposal_list, img_metas, rescale=rescale)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_GDL, self).show_result(data, result, **kwargs)
