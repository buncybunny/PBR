# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList
import torchvision.transforms as transforms
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from mmdet.core.post_processing.img_transforms import img_transforms
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from ..builder import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def th_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        bbox_coors = [res.bboxes for res in sampling_results]
        rois = bbox2roi(bbox_coors)

        bbox_results = self._bbox_forward(stage, x, rois)
        # iou_calculator= build_iou_calculator(dict(type='BboxOverlaps2D'))
        # overlaps = [iou_calculator(gt_bbox, bbox_coors) for gt_bbox in gt_bboxes]
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      img=None,
                      backbone=None,
                      neck=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        # novel_proposals_coors = [[] for _ in range(len(gt_labels))]
        # novel_proposals_coors = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            color_jitter = transforms.ColorJitter(brightness=0.1, hue=0.1)
            gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    # gt_nums_per_img = [sr.pos_gt_bboxes.unique(dim=0).shape[0] for sr in sampling_results]
                    #
                    # gt_inds_ = [sr.pos_inds[sr.pos_is_gt] for sr in sampling_results]
                    # wo_gt_roi_labels = copy.deepcopy(roi_labels)
                    # wo_gt_roi_labelss = []
                    # for ii in range(len(gt_inds_)):
                    #     wo_gt_roi_label = wo_gt_roi_labels[512*ii:512*(ii+1)]
                    #     wo_gt_roi_labelss.append(self.th_delete(wo_gt_roi_label, gt_inds_[ii]))
                    #
                    # # roi_labels_without_gt (512-k)
                    # wo_gt_roi_labels = []
                    # wo_gt_roi_labels.append(roi_labels[gt_nums_per_img[0]:roi_labels.shape[0] // 2])
                    # wo_gt_roi_labels.append(roi_labels[roi_labels.shape[0] // 2 + gt_nums_per_img[1]:])

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)

                    # refine_bboxes에서 gt filtering하고 bbox refine 한 번 더 해줌
                    # before this code: proposal_list{list of 2000 tensors each image}
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

                #     novel_proposals_inds = []
                #     # novel_proposal_list = []
                #     novel_proposal_list = [[] for _ in range(num_imgs)]
                #     novel_gt_labels = [[] for _ in range(num_imgs)]
                #     novel_gt_inds = [[] for _ in range(num_imgs)]
                #     novel_gt_bboxes = []
                #
                #     # novel_proposal_inds: gt제외하고 확실하게 novel로 classify된 애들의 index
                #     for wo_gt_roi_label in wo_gt_roi_labels:
                #         novel_proposals_inds.append([i for i in range(len(wo_gt_roi_label)) if
                #                                      wo_gt_roi_label[i] in range(15, 20)])
                #
                #     for ii, gt_label in enumerate(gt_labels):
                #         for ind, label in enumerate(gt_label):
                #             if label in range(15, 20):
                #                 novel_gt_labels[ii].append(label)
                #                 novel_gt_inds[ii].append(ind)
                #         if len(novel_gt_labels[ii]) != 0:
                #             novel_gt_labels[ii] = torch.stack(novel_gt_labels[ii])
                #
                #     if len(novel_gt_labels[0]) != 0 and len(novel_gt_labels[1]) != 0:
                #         for o, gt_inds in enumerate(novel_gt_inds):
                #             novel_gt_bboxes.append(gt_bboxes[o][gt_inds])
                #
                #         for ii, proposal in enumerate(proposal_list):
                #             novel_proposal_ind = novel_proposals_inds[ii]
                #             for npi in novel_proposal_ind:
                #                 try:
                #                     novel_proposal_list[ii].append(proposal[npi])
                #                 except:
                #                     print("stage", i)
                #                     print("index", npi)
                #                     print("len", len(proposal))
                #                     # print(proposal)
                #             if len(novel_proposal_list[ii]) != 0:
                #                 novel_proposal_list[ii] = torch.stack(novel_proposal_list[ii])
                #                 # novel_proposal_list.append(proposal[novel_proposal_ind])
                #
                # if len(novel_gt_labels[0]) != 0 and len(novel_gt_labels[1]) != 0 and len(
                #         novel_proposal_list[0]) != 0 and len(novel_proposal_list[1]) != 0:
                #     # TODO
                #     test_features = [neck(backbone(color_jitter(img))), neck(backbone(gaussian_blur(img)))]
                #
                #     test_sampling_results = []
                #     for jj in range(num_imgs):
                #         test_feats = test_features[jj]
                #         for j in range(num_imgs):
                #             test_assign_result = bbox_assigner.assign(
                #                 novel_proposal_list[j], novel_gt_bboxes[j], gt_bboxes_ignore[j],
                #                 novel_gt_labels[j])
                #             test_sampling_result = bbox_sampler.sample(
                #                 test_assign_result,
                #                 novel_proposal_list[j],
                #                 novel_gt_bboxes[j],
                #                 novel_gt_labels[j],
                #                 feats=[lvl_feat[j][None] for lvl_feat in test_feats])
                #             test_sampling_results.append(test_sampling_result)
                #     print("aug")
                #     aug_sampling_results = [test_sampling_results[:2], test_sampling_results[2:]]
                #     for r in range(num_imgs):
                #         test_feats = test_features[r]
                #         test_bbox_results = self._bbox_forward_train(i, test_feats, aug_sampling_results[r],
                #                                                      novel_gt_bboxes, novel_gt_labels,
                #                                                      rcnn_train_cfg)
                #         loss_avg_factor = 0.1
                #         for name, value in test_bbox_results['loss_bbox'].items():
                #             losses[f'aug{i}_{r}.{name}'] = (
                #                 value * lw * loss_avg_factor if 'loss' in name else value)

        return losses

    def get_loss_for_aug(self, i, augmented_img_features, augmented_GT_bboxes,
                         novel_gt_labels, rcnn_train_cfg, losses, lw, final_aug_sampling_results, empty, non_empty):

        # TODO:batch size가 2가 아닌 경우
        aug_bbox_results = []
        assert empty is None
        # if empty is not None:
        #     # 빈 곳 메워주는 작업
        #     for j, aug_version in enumerate(augmented_img_features):
        #         augmented_img_features[j] = list(augmented_img_features[j])  # tuple to list
        #         for k, single in enumerate(aug_version):
        #             augmented_img_features[j][k] = torch.cat((single, single), )
        #
        #     augmented_GT_bboxes[empty] = augmented_GT_bboxes[non_empty]
        #     novel_gt_labels[empty] = novel_gt_labels[non_empty]

        for img_feat, sampling_result in zip(augmented_img_features, final_aug_sampling_results):
            aug_bbox_results.append(self._bbox_forward_train(i, img_feat,
                                                             sampling_result,
                                                             augmented_GT_bboxes, novel_gt_labels,
                                                             rcnn_train_cfg))

        loss_avg_factor = 0.1
        for j, aug_bbox_result in enumerate(aug_bbox_results):
            for name, value in aug_bbox_result['loss_bbox'].items():
                losses[f'aug{i}_{j}.{name}'] = (
                    value * lw * loss_avg_factor if 'loss' in name else value)
        return losses

    # def aug_bboxes(self, selected_gt_novel_bboxes, gt_labels):
    #     bbox_gt_labels = []
    #     augmented_bboxes = []
    #     for gt_label in gt_labels:
    #         bbox_gt_labels.append([label for label in gt_label if label in range(15, 20)])
    #     for i, (bbox, label) in enumerate(zip(selected_gt_novel_bboxes, bbox_gt_labels)):
    #         label = torch.tensor(label).to('cuda')
    #         if bbox.shape[0] != 0 and label.shape[0] != 0:
    #             if label.shape[0] == 1:
    #                 label = label.unsqueeze(1)
    #             selected_gt_novel_bboxes[i] = torch.cat((bbox, label), 1)
    #     # augmented_bboxes = [alb_transforms(img[k], selected_gt_novel_bbox) for k, selected_gt_novel_bbox in
    #     #              enumerate(selected_gt_novel_bboxes) if selected_gt_novel_bbox.shape[0] != 0]
    #     return augmented_bboxes

    def aug_imgs(self, selected_gt_novel_bboxes, img, img_metas, num_augs):
        augmented_img = []
        for k, selected_gt_novel_bbox in enumerate(selected_gt_novel_bboxes):
            if selected_gt_novel_bbox.shape[0] != 0:
                augmented_img.append(img_transforms(img[k], num_augs))
            # else:
            #     augmented_img.append(torch.tensor([]))
        # augmented_img = [img_transforms(img[k]) for k, selected_gt_novel_bbox in enumerate(selected_gt_novel_bboxes)
        #                  if selected_gt_novel_bbox.shape[0] != 0]
        return augmented_img

    def aug_bboxes(self, feats, img_metas, proposal_list):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            img_shape = img_meta['img_shape']
            scale_factor = None
            flip = True
            flip_direction = 'horizontal'
            # TODO more flexible
            if proposal_list[i].shape[0] != 0:
                aug_bboxes.append(bbox_mapping(proposal_list[i][:, :4], img_shape,
                                               scale_factor, flip, flip_direction))
            else:
                aug_bboxes.append(torch.tensor([]))
        return aug_bboxes
        #     rois = bbox2roi([proposals])
        #     bbox_results = self._bbox_forward(x, rois)
        #     bboxes, scores = self.bbox_head.get_bboxes(
        #         rois,
        #         bbox_results['cls_score'],
        #         bbox_results['bbox_pred'],
        #         img_shape,
        #         scale_factor,
        #         rescale=False,
        #         cfg=None)
        #     aug_bboxes.append(bboxes)
        #     aug_scores.append(scores)
        # # after merging, bboxes will be rescaled to the original image size
        # merged_bboxes, merged_scores = merge_aug_bboxes(
        #     aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        # if merged_bboxes.shape[0] == 0:
        #     # There is no proposal in the single image
        #     det_bboxes = merged_bboxes.new_zeros(0, 5)
        #     det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)
        # else:
        #     det_bboxes, det_labels = multiclass_nms(merged_bboxes,
        #                                             merged_scores,
        #                                             rcnn_test_cfg.score_thr,
        #                                             rcnn_test_cfg.nms,
        #                                             rcnn_test_cfg.max_per_img)
        # return det_bboxes, det_labels

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in mask_pred
                    ])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = np.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=dummy_scale_factor,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

    def onnx_export(self, x, proposals, img_metas):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals.shape[0] == 1, 'Only support one input image ' \
                                        'while in exporting to ONNX'
        # remove the scores
        rois = proposals[..., :-1]
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        # Eliminate the batch dimension
        rois = rois.view(-1, 4)

        # add dummy batch index
        rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

        max_shape = img_metas[0]['img_shape_for_onnx']
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            # Recover the batch dimension
            rois = rois.reshape(batch_size, num_proposals_per_img,
                                rois.size(-1))
            cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                          cls_score.size(-1))
            bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois = self.bbox_head[i].bbox_coder.decode(
                    rois[..., 1:], bbox_pred, max_shape=max_shape)
                rois = new_rois.reshape(-1, new_rois.shape[-1])
                # add dummy batch index
                rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois],
                                 dim=-1)

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        det_bboxes, det_labels = self.bbox_head[-1].onnx_export(
            rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            batch_index = torch.arange(
                det_bboxes.size(0),
                device=det_bboxes.device).float().view(-1, 1, 1).expand(
                det_bboxes.size(0), det_bboxes.size(1), 1)
            rois = det_bboxes[..., :4]
            mask_rois = torch.cat([batch_index, rois], dim=-1)
            mask_rois = mask_rois.view(-1, 5)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                aug_masks.append(mask_pred)
            max_shape = img_metas[0]['img_shape_for_onnx']
            # calculate the mean of masks from several stage
            mask_pred = sum(aug_masks) / len(aug_masks)
            segm_results = self.mask_head[-1].onnx_export(
                mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
                self.test_cfg, max_shape)
            segm_results = segm_results.reshape(batch_size,
                                                det_bboxes.shape[1],
                                                max_shape[0], max_shape[1])
            return det_bboxes, det_labels, segm_results

# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
#
# import numpy as np
# import torch
# import torch.nn as nn
# from mmcv.runner import ModuleList
# import torchvision.transforms as transforms
# from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
#                         build_sampler, merge_aug_bboxes, merge_aug_masks,
#                         multiclass_nms)
# from mmdet.core.post_processing.img_transforms import img_transforms
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin
# from ..builder import HEADS, build_head, build_roi_extractor
#
#
# @HEADS.register_module()
# class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
#     """Cascade roi head including one bbox head and one mask head.
#
#     https://arxiv.org/abs/1712.00726
#     """
#
#     def __init__(self,
#                  num_stages,
#                  stage_loss_weights,
#                  bbox_roi_extractor=None,
#                  bbox_head=None,
#                  mask_roi_extractor=None,
#                  mask_head=None,
#                  shared_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None,
#                  init_cfg=None):
#         assert bbox_roi_extractor is not None
#         assert bbox_head is not None
#         assert shared_head is None, \
#             'Shared head is not supported in Cascade RCNN anymore'
#
#         self.num_stages = num_stages
#         self.stage_loss_weights = stage_loss_weights
#         super(CascadeRoIHead, self).__init__(
#             bbox_roi_extractor=bbox_roi_extractor,
#             bbox_head=bbox_head,
#             mask_roi_extractor=mask_roi_extractor,
#             mask_head=mask_head,
#             shared_head=shared_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             pretrained=pretrained,
#             init_cfg=init_cfg)
#
#     def init_bbox_head(self, bbox_roi_extractor, bbox_head):
#         """Initialize box head and box roi extractor.
#
#         Args:
#             bbox_roi_extractor (dict): Config of box roi extractor.
#             bbox_head (dict): Config of box in box head.
#         """
#         self.bbox_roi_extractor = ModuleList()
#         self.bbox_head = ModuleList()
#         if not isinstance(bbox_roi_extractor, list):
#             bbox_roi_extractor = [
#                 bbox_roi_extractor for _ in range(self.num_stages)
#             ]
#         if not isinstance(bbox_head, list):
#             bbox_head = [bbox_head for _ in range(self.num_stages)]
#         assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
#         for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
#             self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
#             self.bbox_head.append(build_head(head))
#
#     def init_mask_head(self, mask_roi_extractor, mask_head):
#         """Initialize mask head and mask roi extractor.
#
#         Args:
#             mask_roi_extractor (dict): Config of mask roi extractor.
#             mask_head (dict): Config of mask in mask head.
#         """
#         self.mask_head = nn.ModuleList()
#         if not isinstance(mask_head, list):
#             mask_head = [mask_head for _ in range(self.num_stages)]
#         assert len(mask_head) == self.num_stages
#         for head in mask_head:
#             self.mask_head.append(build_head(head))
#         if mask_roi_extractor is not None:
#             self.share_roi_extractor = False
#             self.mask_roi_extractor = ModuleList()
#             if not isinstance(mask_roi_extractor, list):
#                 mask_roi_extractor = [
#                     mask_roi_extractor for _ in range(self.num_stages)
#                 ]
#             assert len(mask_roi_extractor) == self.num_stages
#             for roi_extractor in mask_roi_extractor:
#                 self.mask_roi_extractor.append(
#                     build_roi_extractor(roi_extractor))
#         else:
#             self.share_roi_extractor = True
#             self.mask_roi_extractor = self.bbox_roi_extractor
#
#     def init_assigner_sampler(self):
#         """Initialize assigner and sampler for each stage."""
#         self.bbox_assigner = []
#         self.bbox_sampler = []
#         if self.train_cfg is not None:
#             for idx, rcnn_train_cfg in enumerate(self.train_cfg):
#                 self.bbox_assigner.append(
#                     build_assigner(rcnn_train_cfg.assigner))
#                 self.current_stage = idx
#                 self.bbox_sampler.append(
#                     build_sampler(rcnn_train_cfg.sampler, context=self))
#
#     def forward_dummy(self, x, proposals):
#         """Dummy forward function."""
#         # bbox head
#         outs = ()
#         rois = bbox2roi([proposals])
#         if self.with_bbox:
#             for i in range(self.num_stages):
#                 bbox_results = self._bbox_forward(i, x, rois)
#                 outs = outs + (bbox_results['cls_score'],
#                                bbox_results['bbox_pred'])
#         # mask heads
#         if self.with_mask:
#             mask_rois = rois[:100]
#             for i in range(self.num_stages):
#                 mask_results = self._mask_forward(i, x, mask_rois)
#                 outs = outs + (mask_results['mask_pred'],)
#         return outs
#
#     def _bbox_forward(self, stage, x, rois):
#         """Box head forward function used in both training and testing."""
#         bbox_roi_extractor = self.bbox_roi_extractor[stage]
#         bbox_head = self.bbox_head[stage]
#         bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
#                                         rois)
#         # do not support caffe_c4 model anymore
#         cls_score, bbox_pred = bbox_head(bbox_feats)
#
#         bbox_results = dict(
#             cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
#         return bbox_results
#
#     def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
#                             gt_labels, rcnn_train_cfg):
#         """Run forward function and calculate loss for box head in training."""
#         bbox_coors = [res.bboxes for res in sampling_results]
#         rois = bbox2roi(bbox_coors)
#
#         bbox_results = self._bbox_forward(stage, x, rois)
#         # iou_calculator= build_iou_calculator(dict(type='BboxOverlaps2D'))
#         # overlaps = [iou_calculator(gt_bbox, bbox_coors) for gt_bbox in gt_bboxes]
#         bbox_targets = self.bbox_head[stage].get_targets(
#             sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
#
#         loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
#                                                bbox_results['bbox_pred'], rois,
#                                                *bbox_targets)
#
#         bbox_results.update(
#             loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
#         return bbox_results
#
#     def _mask_forward(self, stage, x, rois):
#         """Mask head forward function used in both training and testing."""
#         mask_roi_extractor = self.mask_roi_extractor[stage]
#         mask_head = self.mask_head[stage]
#         mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
#                                         rois)
#         # do not support caffe_c4 model anymore
#         mask_pred = mask_head(mask_feats)
#
#         mask_results = dict(mask_pred=mask_pred)
#         return mask_results
#
#     def _mask_forward_train(self,
#                             stage,
#                             x,
#                             sampling_results,
#                             gt_masks,
#                             rcnn_train_cfg,
#                             bbox_feats=None):
#         """Run forward function and calculate loss for mask head in
#         training."""
#         pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
#         mask_results = self._mask_forward(stage, x, pos_rois)
#
#         mask_targets = self.mask_head[stage].get_targets(
#             sampling_results, gt_masks, rcnn_train_cfg)
#         pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
#         loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
#                                                mask_targets, pos_labels)
#
#         mask_results.update(loss_mask=loss_mask)
#         return mask_results
#
#     def forward_train(self,
#                       x,
#                       img_metas,
#                       proposal_list,
#                       gt_bboxes,
#                       gt_labels,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None,
#                       img=None,
#                       backbone=None,
#                       neck=None):
#         """
#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmdet/datasets/pipelines/formatting.py:Collect`.
#             proposals (list[Tensors]): list of region proposals.
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss.
#             gt_masks (None | Tensor) : true segmentation masks for each box
#                 used if the architecture supports a segmentation task.
#
#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         losses = dict()
#         # novel_proposals_coors = [[] for _ in range(len(gt_labels))]
#         # novel_proposals_coors = None
#         for i in range(self.num_stages):
#             self.current_stage = i
#             rcnn_train_cfg = self.train_cfg[i]
#             lw = self.stage_loss_weights[i]
#
#             # assign gts and sample proposals
#             sampling_results = []
#             if self.with_bbox or self.with_mask:
#                 bbox_assigner = self.bbox_assigner[i]
#                 bbox_sampler = self.bbox_sampler[i]
#                 num_imgs = len(img_metas)
#                 if gt_bboxes_ignore is None:
#                     gt_bboxes_ignore = [None for _ in range(num_imgs)]
#
#                 for j in range(num_imgs):
#                     assign_result = bbox_assigner.assign(
#                         proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
#                         gt_labels[j])
#                     sampling_result = bbox_sampler.sample(
#                         assign_result,
#                         proposal_list[j],
#                         gt_bboxes[j],
#                         gt_labels[j],
#                         feats=[lvl_feat[j][None] for lvl_feat in x])
#                     sampling_results.append(sampling_result)
#
#             # bbox head forward and loss
#             bbox_results = self._bbox_forward_train(i, x, sampling_results,
#                                                     gt_bboxes, gt_labels,
#                                                     rcnn_train_cfg)
#
#             for name, value in bbox_results['loss_bbox'].items():
#                 losses[f's{i}.{name}'] = (
#                     value * lw if 'loss' in name else value)
#
#             color_jitter = transforms.ColorJitter(brightness=0.1, hue=0.1)
#             gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
#
#             # refine bboxes
#             if i < self.num_stages - 1:
#                 pos_is_gts = [res.pos_is_gt for res in sampling_results]
#                 # bbox_targets is a tuple
#                 roi_labels = bbox_results['bbox_targets'][0]
#                 with torch.no_grad():
#                     cls_score = bbox_results['cls_score']
#                     if self.bbox_head[i].custom_activation:
#                         cls_score = self.bbox_head[i].loss_cls.get_activation(
#                             cls_score)
#
#                     # Empty proposal.
#                     if cls_score.numel() == 0:
#                         break
#                     gt_nums_per_img = [sr.pos_gt_bboxes.unique(dim=0).shape[0] for sr in sampling_results]
#
#                     # roi_labels_without_gt (512-k)
#                     wo_gt_roi_labels = []
#                     wo_gt_roi_labels.append(roi_labels[gt_nums_per_img[0]:roi_labels.shape[0] // 2])
#                     wo_gt_roi_labels.append(roi_labels[roi_labels.shape[0] // 2 + gt_nums_per_img[1]:])
#
#                     roi_labels = torch.where(
#                         roi_labels == self.bbox_head[i].num_classes,
#                         cls_score[:, :-1].argmax(1), roi_labels)
#
#                     # refine_bboxes에서 gt filtering하고 bbox refine 한 번 더 해줌
#                     proposal_list = self.bbox_head[i].refine_bboxes(
#                         bbox_results['rois'], roi_labels,
#                         bbox_results['bbox_pred'], pos_is_gts, img_metas)
#
#         return losses
#
#     def get_loss_for_aug(self, i, augmented_img_features, augmented_GT_bboxes,
#                          novel_gt_labels, rcnn_train_cfg, losses, lw, final_aug_sampling_results, empty, non_empty):
#
#         # TODO:batch size가 2가 아닌 경우
#         aug_bbox_results = []
#         assert empty is None
#         # if empty is not None:
#         #     # 빈 곳 메워주는 작업
#         #     for j, aug_version in enumerate(augmented_img_features):
#         #         augmented_img_features[j] = list(augmented_img_features[j])  # tuple to list
#         #         for k, single in enumerate(aug_version):
#         #             augmented_img_features[j][k] = torch.cat((single, single), )
#         #
#         #     augmented_GT_bboxes[empty] = augmented_GT_bboxes[non_empty]
#         #     novel_gt_labels[empty] = novel_gt_labels[non_empty]
#
#         for img_feat, sampling_result in zip(augmented_img_features, final_aug_sampling_results):
#             aug_bbox_results.append(self._bbox_forward_train(i, img_feat,
#                                                              sampling_result,
#                                                              augmented_GT_bboxes, novel_gt_labels,
#                                                              rcnn_train_cfg))
#
#         loss_avg_factor = 0.1
#         for j, aug_bbox_result in enumerate(aug_bbox_results):
#             for name, value in aug_bbox_result['loss_bbox'].items():
#                 losses[f'aug{i}_{j}.{name}'] = (
#                     value * lw * loss_avg_factor if 'loss' in name else value)
#         return losses
#
#     # def aug_bboxes(self, selected_gt_novel_bboxes, gt_labels):
#     #     bbox_gt_labels = []
#     #     augmented_bboxes = []
#     #     for gt_label in gt_labels:
#     #         bbox_gt_labels.append([label for label in gt_label if label in range(15, 20)])
#     #     for i, (bbox, label) in enumerate(zip(selected_gt_novel_bboxes, bbox_gt_labels)):
#     #         label = torch.tensor(label).to('cuda')
#     #         if bbox.shape[0] != 0 and label.shape[0] != 0:
#     #             if label.shape[0] == 1:
#     #                 label = label.unsqueeze(1)
#     #             selected_gt_novel_bboxes[i] = torch.cat((bbox, label), 1)
#     #     # augmented_bboxes = [alb_transforms(img[k], selected_gt_novel_bbox) for k, selected_gt_novel_bbox in
#     #     #              enumerate(selected_gt_novel_bboxes) if selected_gt_novel_bbox.shape[0] != 0]
#     #     return augmented_bboxes
#
#     def aug_imgs(self, selected_gt_novel_bboxes, img, img_metas, num_augs):
#         augmented_img = []
#         for k, selected_gt_novel_bbox in enumerate(selected_gt_novel_bboxes):
#             if selected_gt_novel_bbox.shape[0] != 0:
#                 augmented_img.append(img_transforms(img[k], num_augs))
#             # else:
#             #     augmented_img.append(torch.tensor([]))
#         # augmented_img = [img_transforms(img[k]) for k, selected_gt_novel_bbox in enumerate(selected_gt_novel_bboxes)
#         #                  if selected_gt_novel_bbox.shape[0] != 0]
#         return augmented_img
#
#     def aug_bboxes(self, feats, img_metas, proposal_list):
#         """Test det bboxes with test time augmentation."""
#         aug_bboxes = []
#         for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
#             # only one image in the batch
#             img_shape = img_meta['img_shape']
#             scale_factor = None
#             flip = True
#             flip_direction = 'horizontal'
#             # TODO more flexible
#             if proposal_list[i].shape[0] != 0:
#                 aug_bboxes.append(bbox_mapping(proposal_list[i][:, :4], img_shape,
#                                                scale_factor, flip, flip_direction))
#             else:
#                 aug_bboxes.append(torch.tensor([]))
#         return aug_bboxes
#         #     rois = bbox2roi([proposals])
#         #     bbox_results = self._bbox_forward(x, rois)
#         #     bboxes, scores = self.bbox_head.get_bboxes(
#         #         rois,
#         #         bbox_results['cls_score'],
#         #         bbox_results['bbox_pred'],
#         #         img_shape,
#         #         scale_factor,
#         #         rescale=False,
#         #         cfg=None)
#         #     aug_bboxes.append(bboxes)
#         #     aug_scores.append(scores)
#         # # after merging, bboxes will be rescaled to the original image size
#         # merged_bboxes, merged_scores = merge_aug_bboxes(
#         #     aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
#         # if merged_bboxes.shape[0] == 0:
#         #     # There is no proposal in the single image
#         #     det_bboxes = merged_bboxes.new_zeros(0, 5)
#         #     det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)
#         # else:
#         #     det_bboxes, det_labels = multiclass_nms(merged_bboxes,
#         #                                             merged_scores,
#         #                                             rcnn_test_cfg.score_thr,
#         #                                             rcnn_test_cfg.nms,
#         #                                             rcnn_test_cfg.max_per_img)
#         # return det_bboxes, det_labels
#
#     def simple_test(self, x, proposal_list, img_metas, rescale=False):
#         """Test without augmentation.
#
#         Args:
#             x (tuple[Tensor]): Features from upstream network. Each
#                 has shape (batch_size, c, h, w).
#             proposal_list (list(Tensor)): Proposals from rpn head.
#                 Each has shape (num_proposals, 5), last dimension
#                 5 represent (x1, y1, x2, y2, score).
#             img_metas (list[dict]): Meta information of images.
#             rescale (bool): Whether to rescale the results to
#                 the original image. Default: True.
#
#         Returns:
#             list[list[np.ndarray]] or list[tuple]: When no mask branch,
#             it is bbox results of each image and classes with type
#             `list[list[np.ndarray]]`. The outer list
#             corresponds to each image. The inner list
#             corresponds to each class. When the model has mask branch,
#             it contains bbox results and mask results.
#             The outer list corresponds to each image, and first element
#             of tuple is bbox results, second element is mask results.
#         """
#         assert self.with_bbox, 'Bbox head must be implemented.'
#         num_imgs = len(proposal_list)
#         img_shapes = tuple(meta['img_shape'] for meta in img_metas)
#         ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
#         scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
#
#         # "ms" in variable names means multi-stage
#         ms_bbox_result = {}
#         ms_segm_result = {}
#         ms_scores = []
#         rcnn_test_cfg = self.test_cfg
#
#         rois = bbox2roi(proposal_list)
#
#         if rois.shape[0] == 0:
#             # There is no proposal in the whole batch
#             bbox_results = [[
#                 np.zeros((0, 5), dtype=np.float32)
#                 for _ in range(self.bbox_head[-1].num_classes)
#             ]] * num_imgs
#
#             if self.with_mask:
#                 mask_classes = self.mask_head[-1].num_classes
#                 segm_results = [[[] for _ in range(mask_classes)]
#                                 for _ in range(num_imgs)]
#                 results = list(zip(bbox_results, segm_results))
#             else:
#                 results = bbox_results
#
#             return results
#
#         for i in range(self.num_stages):
#             bbox_results = self._bbox_forward(i, x, rois)
#
#             # split batch bbox prediction back to each image
#             cls_score = bbox_results['cls_score']
#             bbox_pred = bbox_results['bbox_pred']
#             num_proposals_per_img = tuple(
#                 len(proposals) for proposals in proposal_list)
#             rois = rois.split(num_proposals_per_img, 0)
#             cls_score = cls_score.split(num_proposals_per_img, 0)
#             if isinstance(bbox_pred, torch.Tensor):
#                 bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
#             else:
#                 bbox_pred = self.bbox_head[i].bbox_pred_split(
#                     bbox_pred, num_proposals_per_img)
#             ms_scores.append(cls_score)
#
#             if i < self.num_stages - 1:
#                 if self.bbox_head[i].custom_activation:
#                     cls_score = [
#                         self.bbox_head[i].loss_cls.get_activation(s)
#                         for s in cls_score
#                     ]
#                 refine_rois_list = []
#                 for j in range(num_imgs):
#                     if rois[j].shape[0] > 0:
#                         bbox_label = cls_score[j][:, :-1].argmax(dim=1)
#                         refined_rois = self.bbox_head[i].regress_by_class(
#                             rois[j], bbox_label, bbox_pred[j], img_metas[j])
#                         refine_rois_list.append(refined_rois)
#                 rois = torch.cat(refine_rois_list)
#
#         # average scores of each image by stages
#         cls_score = [
#             sum([score[i] for score in ms_scores]) / float(len(ms_scores))
#             for i in range(num_imgs)
#         ]
#
#         # apply bbox post-processing to each image individually
#         det_bboxes = []
#         det_labels = []
#         for i in range(num_imgs):
#             det_bbox, det_label = self.bbox_head[-1].get_bboxes(
#                 rois[i],
#                 cls_score[i],
#                 bbox_pred[i],
#                 img_shapes[i],
#                 scale_factors[i],
#                 rescale=rescale,
#                 cfg=rcnn_test_cfg)
#             det_bboxes.append(det_bbox)
#             det_labels.append(det_label)
#
#         bbox_results = [
#             bbox2result(det_bboxes[i], det_labels[i],
#                         self.bbox_head[-1].num_classes)
#             for i in range(num_imgs)
#         ]
#         ms_bbox_result['ensemble'] = bbox_results
#
#         if self.with_mask:
#             if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
#                 mask_classes = self.mask_head[-1].num_classes
#                 segm_results = [[[] for _ in range(mask_classes)]
#                                 for _ in range(num_imgs)]
#             else:
#                 if rescale and not isinstance(scale_factors[0], float):
#                     scale_factors = [
#                         torch.from_numpy(scale_factor).to(det_bboxes[0].device)
#                         for scale_factor in scale_factors
#                     ]
#                 _bboxes = [
#                     det_bboxes[i][:, :4] *
#                     scale_factors[i] if rescale else det_bboxes[i][:, :4]
#                     for i in range(len(det_bboxes))
#                 ]
#                 mask_rois = bbox2roi(_bboxes)
#                 num_mask_rois_per_img = tuple(
#                     _bbox.size(0) for _bbox in _bboxes)
#                 aug_masks = []
#                 for i in range(self.num_stages):
#                     mask_results = self._mask_forward(i, x, mask_rois)
#                     mask_pred = mask_results['mask_pred']
#                     # split batch mask prediction back to each image
#                     mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
#                     aug_masks.append([
#                         m.sigmoid().cpu().detach().numpy() for m in mask_pred
#                     ])
#
#                 # apply mask post-processing to each image individually
#                 segm_results = []
#                 for i in range(num_imgs):
#                     if det_bboxes[i].shape[0] == 0:
#                         segm_results.append(
#                             [[]
#                              for _ in range(self.mask_head[-1].num_classes)])
#                     else:
#                         aug_mask = [mask[i] for mask in aug_masks]
#                         merged_masks = merge_aug_masks(
#                             aug_mask, [[img_metas[i]]] * self.num_stages,
#                             rcnn_test_cfg)
#                         segm_result = self.mask_head[-1].get_seg_masks(
#                             merged_masks, _bboxes[i], det_labels[i],
#                             rcnn_test_cfg, ori_shapes[i], scale_factors[i],
#                             rescale)
#                         segm_results.append(segm_result)
#             ms_segm_result['ensemble'] = segm_results
#
#         if self.with_mask:
#             results = list(
#                 zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
#         else:
#             results = ms_bbox_result['ensemble']
#
#         return results
#
#     def aug_test(self, features, proposal_list, img_metas, rescale=False):
#         """Test with augmentations.
#
#         If rescale is False, then returned bboxes and masks will fit the scale
#         of imgs[0].
#         """
#         rcnn_test_cfg = self.test_cfg
#         aug_bboxes = []
#         aug_scores = []
#         for x, img_meta in zip(features, img_metas):
#             # only one image in the batch
#             img_shape = img_meta[0]['img_shape']
#             scale_factor = img_meta[0]['scale_factor']
#             flip = img_meta[0]['flip']
#             flip_direction = img_meta[0]['flip_direction']
#
#             proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
#                                      scale_factor, flip, flip_direction)
#             # "ms" in variable names means multi-stage
#             ms_scores = []
#
#             rois = bbox2roi([proposals])
#
#             if rois.shape[0] == 0:
#                 # There is no proposal in the single image
#                 aug_bboxes.append(rois.new_zeros(0, 4))
#                 aug_scores.append(rois.new_zeros(0, 1))
#                 continue
#
#             for i in range(self.num_stages):
#                 bbox_results = self._bbox_forward(i, x, rois)
#                 ms_scores.append(bbox_results['cls_score'])
#
#                 if i < self.num_stages - 1:
#                     cls_score = bbox_results['cls_score']
#                     if self.bbox_head[i].custom_activation:
#                         cls_score = self.bbox_head[i].loss_cls.get_activation(
#                             cls_score)
#                     bbox_label = cls_score[:, :-1].argmax(dim=1)
#                     rois = self.bbox_head[i].regress_by_class(
#                         rois, bbox_label, bbox_results['bbox_pred'],
#                         img_meta[0])
#
#             cls_score = sum(ms_scores) / float(len(ms_scores))
#             bboxes, scores = self.bbox_head[-1].get_bboxes(
#                 rois,
#                 cls_score,
#                 bbox_results['bbox_pred'],
#                 img_shape,
#                 scale_factor,
#                 rescale=False,
#                 cfg=None)
#             aug_bboxes.append(bboxes)
#             aug_scores.append(scores)
#
#         # after merging, bboxes will be rescaled to the original image size
#         merged_bboxes, merged_scores = merge_aug_bboxes(
#             aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
#         det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
#                                                 rcnn_test_cfg.score_thr,
#                                                 rcnn_test_cfg.nms,
#                                                 rcnn_test_cfg.max_per_img)
#
#         bbox_result = bbox2result(det_bboxes, det_labels,
#                                   self.bbox_head[-1].num_classes)
#
#         if self.with_mask:
#             if det_bboxes.shape[0] == 0:
#                 segm_result = [[]
#                                for _ in range(self.mask_head[-1].num_classes)]
#             else:
#                 aug_masks = []
#                 aug_img_metas = []
#                 for x, img_meta in zip(features, img_metas):
#                     img_shape = img_meta[0]['img_shape']
#                     scale_factor = img_meta[0]['scale_factor']
#                     flip = img_meta[0]['flip']
#                     flip_direction = img_meta[0]['flip_direction']
#                     _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
#                                            scale_factor, flip, flip_direction)
#                     mask_rois = bbox2roi([_bboxes])
#                     for i in range(self.num_stages):
#                         mask_results = self._mask_forward(i, x, mask_rois)
#                         aug_masks.append(
#                             mask_results['mask_pred'].sigmoid().cpu().numpy())
#                         aug_img_metas.append(img_meta)
#                 merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
#                                                self.test_cfg)
#
#                 ori_shape = img_metas[0][0]['ori_shape']
#                 dummy_scale_factor = np.ones(4)
#                 segm_result = self.mask_head[-1].get_seg_masks(
#                     merged_masks,
#                     det_bboxes,
#                     det_labels,
#                     rcnn_test_cfg,
#                     ori_shape,
#                     scale_factor=dummy_scale_factor,
#                     rescale=False)
#             return [(bbox_result, segm_result)]
#         else:
#             return [bbox_result]
#
#     def onnx_export(self, x, proposals, img_metas):
#         assert self.with_bbox, 'Bbox head must be implemented.'
#         assert proposals.shape[0] == 1, 'Only support one input image ' \
#                                         'while in exporting to ONNX'
#         # remove the scores
#         rois = proposals[..., :-1]
#         batch_size = rois.shape[0]
#         num_proposals_per_img = rois.shape[1]
#         # Eliminate the batch dimension
#         rois = rois.view(-1, 4)
#
#         # add dummy batch index
#         rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)
#
#         max_shape = img_metas[0]['img_shape_for_onnx']
#         ms_scores = []
#         rcnn_test_cfg = self.test_cfg
#
#         for i in range(self.num_stages):
#             bbox_results = self._bbox_forward(i, x, rois)
#
#             cls_score = bbox_results['cls_score']
#             bbox_pred = bbox_results['bbox_pred']
#             # Recover the batch dimension
#             rois = rois.reshape(batch_size, num_proposals_per_img,
#                                 rois.size(-1))
#             cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
#                                           cls_score.size(-1))
#             bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
#             ms_scores.append(cls_score)
#             if i < self.num_stages - 1:
#                 assert self.bbox_head[i].reg_class_agnostic
#                 new_rois = self.bbox_head[i].bbox_coder.decode(
#                     rois[..., 1:], bbox_pred, max_shape=max_shape)
#                 rois = new_rois.reshape(-1, new_rois.shape[-1])
#                 # add dummy batch index
#                 rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois],
#                                  dim=-1)
#
#         cls_score = sum(ms_scores) / float(len(ms_scores))
#         bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
#         rois = rois.reshape(batch_size, num_proposals_per_img, -1)
#         det_bboxes, det_labels = self.bbox_head[-1].onnx_export(
#             rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)
#
#         if not self.with_mask:
#             return det_bboxes, det_labels
#         else:
#             batch_index = torch.arange(
#                 det_bboxes.size(0),
#                 device=det_bboxes.device).float().view(-1, 1, 1).expand(
#                 det_bboxes.size(0), det_bboxes.size(1), 1)
#             rois = det_bboxes[..., :4]
#             mask_rois = torch.cat([batch_index, rois], dim=-1)
#             mask_rois = mask_rois.view(-1, 5)
#             aug_masks = []
#             for i in range(self.num_stages):
#                 mask_results = self._mask_forward(i, x, mask_rois)
#                 mask_pred = mask_results['mask_pred']
#                 aug_masks.append(mask_pred)
#             max_shape = img_metas[0]['img_shape_for_onnx']
#             # calculate the mean of masks from several stage
#             mask_pred = sum(aug_masks) / len(aug_masks)
#             segm_results = self.mask_head[-1].onnx_export(
#                 mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
#                 self.test_cfg, max_shape)
#             segm_results = segm_results.reshape(batch_size,
#                                                 det_bboxes.shape[1],
#                                                 max_shape[0], max_shape[1])
#             return det_bboxes, det_labels, segm_results
