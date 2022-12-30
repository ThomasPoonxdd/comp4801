import torch

import tf_example_decoder

from utils import input_utils

class Parser(object):
    def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               rpn_match_threshold=0.7,
               rpn_unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               max_num_instances=100,
               include_mask=False,
               mask_crop_size=112,
               use_bfloat16=True,
               mode=None,
               # for vild
               visual_feature_distill=False,
               visual_feature_dim=512,
               max_num_rois=300,
               filter_distill_boxes_size=0):

        """Initializes parameters for parsing annotations in the dataset.
        Args:
        output_size: `Tensor` or `list` for [height, width] of output image. The
            output_size should be divided by the largest feature stride 2^max_level.
        min_level: `int` number of minimum level of the output feature pyramid.
        max_level: `int` number of maximum level of the output feature pyramid.
        num_scales: `int` number representing intermediate scales added
            on each level. For instances, num_scales=2 adds one additional
            intermediate anchor scales [2^0, 2^0.5] on each level.
        aspect_ratios: `list` of float numbers representing the aspect raito
            anchors added on each level. The number indicates the ratio of width to
            height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
            on each scale level.
        anchor_size: `float` number representing the scale of size of the base
            anchor to the feature stride 2^level.
        rpn_match_threshold:
        rpn_unmatched_threshold:
        rpn_batch_size_per_im:
        rpn_fg_fraction:
        aug_rand_hflip: `bool`, if True, augment training with random
            horizontal flip.
        aug_scale_min: `float`, the minimum scale applied to `output_size` for
            data augmentation during training.
        aug_scale_max: `float`, the maximum scale applied to `output_size` for
            data augmentation during training.
        skip_crowd_during_training: `bool`, if True, skip annotations labeled with
            `is_crowd` equals to 1.
        max_num_instances: `int` number of maximum number of instances in an
            image. The groundtruth data will be padded to `max_num_instances`.
        include_mask: a bool to indicate whether parse mask groundtruth.
        mask_crop_size: the size which groundtruth mask is cropped to.
        use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
        mode: a ModeKeys. Specifies if this is training, evaluation, prediction
            or prediction with groundtruths in the outputs.
        visual_feature_distill: `bool`, whether to distill visual features.
        visual_feature_dim: `int`, the distillation feature dimension.
        max_num_rois: `int`, the number of precomputed rois to distill.
        filter_distill_boxes_size: `int`, specify the size (height/width) of the
        boxes such that the precomputed rois smaller than this size will be
            filtered out.
        """

        self._mode = mode
        self._max_num_instances = max_num_instances
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = (mode == self.training) # self._is_training = (mode == ModeKeys.TRAIN)

        self._example_decoder = tf_example_decoder.TfExampleDecoder(
            include_mask=include_mask,
            visual_feature_distill=visual_feature_distill)

        # Anchor.
        self._output_size = output_size
        self._min_level = min_level
        self._max_level = max_level
        self._num_scales = num_scales
        self._aspect_ratios = aspect_ratios
        self._anchor_size = anchor_size

        # Target assigning.
        self._rpn_match_threshold = rpn_match_threshold
        self._rpn_unmatched_threshold = rpn_unmatched_threshold
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._rpn_fg_fraction = rpn_fg_fraction

        # Data augmentation.
        self._aug_rand_hflip = aug_rand_hflip
        self._aug_scale_min = aug_scale_min
        self._aug_scale_max = aug_scale_max

        # Mask.
        self._include_mask = include_mask
        self._mask_crop_size = mask_crop_size

        # Device.
        self._use_bfloat16 = use_bfloat16

        # Visual feature distill, only True during training
        self._visual_feature_distill = visual_feature_distill
        self._visual_feature_dim = visual_feature_dim

        if self._visual_feature_distill:
            self._max_num_rois = max_num_rois
            self._filter_distill_boxes_size = filter_distill_boxes_size

        # Data is parsed depending on the model Modekey.
        if self.training == True: # if mode == ModeKeys.TRAIN:
            self._parse_fn = self._parse_train_data
        elif self.training == False: # elif mode == ModeKeys.EVAL:
            self._parse_fn = self._parse_eval_data
        # elif mode == ModeKeys.PREDICT or mode == ModeKeys.PREDICT_WITH_GT:
        #     self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    

    def __call__(self, value):
        """Parses data to an image and associated training labels.
        Args:
        value: a string tensor holding a serialized tf.Example proto.
        Returns:
        image, labels: if mode == ModeKeys.TRAIN. see _parse_train_data.
        {'images': image, 'labels': labels}: if mode == ModeKeys.PREDICT
            or ModeKeys.PREDICT_WITH_GT.
        """
        data = self._example_decoder.decode(
            value,
            visual_feat_dim=self._visual_feature_dim)
        return self._parse_fn(data)



    def _parse_train_data(self, data):
        """Parses data for training.
        Args:
        data: the decoded tensor dictionary from TfExampleDecoder.
        Returns:
        image: image tensor that is preproessed to have normalized value and
            dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following describes
            {key: value} pairs in the dictionary.
            image_info: a 2D `Tensor` that encodes the information of the image and
            the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width],
            anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each level.
            rpn_score_targets: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, anchors_per_location]. The height_l and
            width_l represent the dimension of class logits at l-th level.
            rpn_box_targets: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, anchors_per_location * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
            gt_boxes: Groundtruth bounding box annotations. The box is represented
            in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
            image that is fed to the network. The tennsor is padded with -1 to
            the fixed dimension [self._max_num_instances, 4].
            gt_classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
            gt_masks: groundtrugh masks cropped by the bounding box and
            resized to a fixed size determined by mask_crop_size.
        """
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        if self._include_mask:
            masks = data['groundtruth_instance_masks']
        if self._visual_feature_distill:
            roi_boxes = data['roi_boxes']
        distill_features = data['groundtruth_visual_features']

        is_crowds = data['groundtruth_is_crowd']
        # Skips annotations with `is_crowd` = True.
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = classes.size()[0] # num_groundtrtuhs = tf.shape(classes)[0]
            
            # indices = tf.cond(
            #     tf.greater(tf.size(is_crowds), 0),
            #     lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            #     lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            if (is_crowds.nelement()>0):
                indices = torch.where(torch.logical_not(is_crowds))[0]
            else:
                indices = torch.arange(0, num_groundtrtuhs).to(dtype=torch.int64)

            classes = torch.gather(classes, 1, indices) # classes = tf.gather(classes, indices)
            boxes = torch.gather(boxes, 1, indices) # boxes = tf.gather(boxes, indices)
            if self._include_mask:
                masks = torch.gather(masks, 1, indices) # masks = tf.gather(masks, indices)

        # Gets original image and its size.
        image = data['image']
        image_shape = image.size()[0:2]

        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Flips image randomly during training.
        if self._aug_rand_hflip:
            if self._visual_feature_distill:
                assert self._include_mask
                image, boxes, masks, roi_boxes = input_utils.random_horizontal_flip(
                    image, boxes, masks, roi_boxes)
            if self._include_mask:
                image, boxes, masks = input_utils.random_horizontal_flip(
                    image, boxes, masks)
            else:
                image, boxes = input_utils.random_horizontal_flip(
                    image, boxes)