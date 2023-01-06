# Copyright 2019 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import torch
import torchvision

# def _get_source_id_from_encoded_image(parsed_tensors):
#   return tf.strings.as_string(
#       tf.strings.to_hash_bucket_fast(parsed_tensors['image/encoded'],
#                                      2**63 - 1))


class TfExampleDecoder(object):
  """Tensorflow Example proto decoder."""

  def __init__(
      self,
      include_mask=False,
      # copypara:strip_begin
      include_polygon=False,
      # copypara:strip_end
      regenerate_source_id=False,
      visual_feature_distill=False,
  ):
    self._include_mask = include_mask
    # copypara:strip_begin
    self._include_polygon = include_polygon
    # copypara:strip_end
    self._regenerate_source_id = regenerate_source_id
    self._visual_feature_distill = visual_feature_distill
    self._keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/source_id': tf.FixedLenFeature((), tf.string, ''),
        'image/height': tf.FixedLenFeature((), tf.int64, -1),
        'image/width': tf.FixedLenFeature((), tf.int64, -1),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/area': tf.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.VarLenFeature(tf.int64),
        # ViLD visual feature distill.
        'image/roi/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/roi/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/roi/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/roi/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/roi/bbox/score': tf.VarLenFeature(tf.float32),
        'image/roi/feature': tf.VarLenFeature(tf.float32),
    }
    if include_mask:
      self._keys_to_features['image/object/mask'] = tf.VarLenFeature(tf.string)

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = torchvision.io.decode_image(parsed_tensors['image/encoded'], torchvision.io.ImageReadMode.RGB_ALPHA)

    ### Need translation
    ### pytorch set_shape works differently
    image.set_shape([None, None, 3])
    return image

  def _decode_boxes(self, parsed_tensors, box_key_prefix='image/object/bbox'):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = torch.maximum(parsed_tensors[f'{box_key_prefix}/xmin'], 0)
    xmax = parsed_tensors[f'{box_key_prefix}/xmax']
    ymin = torch.maximum(parsed_tensors[f'{box_key_prefix}/ymin'], 0)
    ymax = parsed_tensors[f'{box_key_prefix}/ymax']
    return torch.stack([ymin, xmin, ymax, xmax], -1)

  def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""
    def _decode_png_mask(png_bytes):
      mask = torch.squeeze(
          torchvision.io.decode_png(png_bytes, torchvision.io.ImageReadMode.GRAY).to(dtpye=torch.unit8), -1)
      mask = mask.to(dtype=torch.float32)

      ### Need translation
      ### pytorch set_shape works differently
      mask.set_shape([None, None])
      return mask

    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    masks = parsed_tensors['image/object/mask']

    if torch.size(masks) > 0:
      return tf.map_fn(_decode_png_mask, masks, dtype=tf.float32)
    else:
      return torch.zeros(0, height, width).to(dtype=torch.float32)


  def _decode_areas(self, parsed_tensors):
    xmin = torch.maximum(parsed_tensors['image/object/bbox/xmin'], 0)
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = torch.maximum(parsed_tensors['image/object/bbox/ymin'], 0)
    ymax = parsed_tensors['image/object/bbox/ymax']
    height = parsed_tensors['image/height'].to(dtpye=torch.float32)
    width = parsed_tensors['image/width'].to(dtpye=torch.float32)

    if parsed_tensors['image/object/area'].size()[0] > 0:
      return parsed_tensors['image/object/area']
    else:
      return (xmax - xmin) * (ymax - ymin) * height * width


  def decode(self, serialized_example, visual_feat_dim=512):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.
      visual_feat_dim: `int`, the dimension of the visual feature to distill.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: an int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    parsed_tensors = tf.io.parse_single_example(
        serialized_example, self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse_tensor_to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse_tensor_to_dense(
              parsed_tensors[k], default_value=0)

    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)

    decode_image_shape = torch.logical_or(
        torch.equal(parsed_tensors['image/height'], -1),
        torch.equal(parsed_tensors['image/width'], -1))
    
    decode_image_shape = torch.logical_or(
        torch.equal(parsed_tensors['image/height'], -1),
        torch.equal(parsed_tensors['image/width'], -1))
    image_shape = image.size().to(dtpye=torch.int64)

    parsed_tensors['image/height'] = torch.where(decode_image_shape,
                                              image_shape[0],
                                              parsed_tensors['image/height'])
    parsed_tensors['image/width'] = torch.where(decode_image_shape, image_shape[1],
                                             parsed_tensors['image/width'])

    if parsed_tensors['image/object/is_crowd'].size()[0] > 0:
      is_crowds = parsed_tensors['image/object/is_crowd'].to(dtype=torch.bool)
    else:
      is_crowds = torch.zeros_like(parsed_tensors['image/object/class/label']).to(dtype=torch.bool)

    if self._regenerate_source_id:
      source_id = _get_source_id_from_encoded_image(parsed_tensors)
    else:
      source_id = tf.cond(
          tf.greater(tf.strings.length(parsed_tensors['image/source_id']),
                     0), lambda: parsed_tensors['image/source_id'],
          lambda: _get_source_id_from_encoded_image(parsed_tensors))
    if self._include_mask:
      masks = self._decode_masks(parsed_tensors)

    decoded_tensors = {
        'image': image,
        'source_id': source_id,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': parsed_tensors['image/object/class/label'],
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    if self._include_mask:
      decoded_tensors.update({
          'groundtruth_instance_masks': masks,
          'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
      })

      if self._visual_feature_distill:
        # [num_roi_boxes, 4]
        roi_boxes = self._decode_boxes(
            parsed_tensors, box_key_prefix='image/roi/bbox')
        # [num_roi_boxes, visual_feat_dim]
        roi_features = torch.reshape(parsed_tensors['image/roi/feature'],
                                  [-1, visual_feat_dim])
        roi_scores = parsed_tensors['image/roi/bbox/score']
        decoded_tensors.update({
            'roi_boxes': roi_boxes,
            'groundtruth_visual_features': roi_features,
            'roi_scores': roi_scores,
        })
    return decoded_tensors

""" For testing the decoder"""
import json
import os
import os.path as osp
import hashlib

# Translated from tpu-master/models/official/detection/projects/vild/preprocessing/create_lvis_tf_record.py
# To test how to pass image info from json to decoder
def testing():
    json_path = '/Users/hokwanchu/Downloads/4801/sorted.json'
    with open(json_path) as json_file:
        data = json.load(json_file)
        for i in data['images']:
            image = i
            id = image['id']
            image_dir = '/Users/hokwanchu/Documents/HKU/Study/BENG_CS/COMP4801/Preprocessing/dataset/train'
            annotations = []
            for anno in data['annotations']:
                if anno['image_id'] == id:
                    annotations.append(anno)
            category_index = data['categories']

            image_height = image['height']
            image_width = image['width']
            filename = image['coco_url']
            filename = osp.join(*filename.split('/')[-2:])

            image_id = image['id']
            image_not_exhaustive_category_ids = image['not_exhaustive_category_ids']
            image_neg_category_ids = image['neg_category_ids']

            full_path = os.path.join(image_dir, filename)

            if not osp.exists(full_path):
                # return False, None, None
                pass

            with open(full_path, 'rb') as fid:
                encoded_jpg = fid.read()

            key = hashlib.sha256(encoded_jpg).hexdigest()
            feature_dict = {
                'image/height':
                    image_height,
                'image/width':
                    image_width,
                'image/filename':
                    filename.encode('utf8'),
                'image/source_id':
                    str(image_id).encode('utf8'),
                'image/key/sha256':
                    key.encode('utf8'),
                'image/encoded':
                    encoded_jpg,
                'image/format':
                    'jpeg'.encode('utf8'),
                'image/not_exhaustive_category_ids':
                    image_not_exhaustive_category_ids,
                'image/image_neg_category_ids':
                    image_neg_category_ids,
            }

            if len(annotations) > 0:
                xmin = []
                xmax = []
                ymin = []
                ymax = []
                is_crowd = []
                category_names = []
                category_ids = []
                area = []
                encoded_mask_png = []
                for object_annotations in annotations:
                    (x, y, width, height) = tuple(object_annotations['bbox'])

                    xmin_single = max(float(x) / image_width, 0.0)
                    xmax_single = min(float(x + width) / image_width, 1.0)
                    ymin_single = max(float(y) / image_height, 0.0)
                    ymax_single = min(float(y + height) / image_height, 1.0)
                    if xmax_single <= xmin_single or ymax_single <= ymin_single:
                        continue
                    xmin.append(xmin_single)
                    xmax.append(xmax_single)
                    ymin.append(ymin_single)
                    ymax.append(ymax_single)

                    is_crowd.append(0)
                    category_id = int(object_annotations['category_id'])
                    category_ids.append(category_id)
                    category_names.append(category_index[category_id]['name'].encode('utf8'))
                    area.append(object_annotations['area'])

                feature_dict.update({
                    'image/object/bbox/xmin':
                        xmin,
                    'image/object/bbox/xmax':
                        xmax,
                    'image/object/bbox/ymin':
                        ymin,
                    'image/object/bbox/ymax':
                        ymax,
                    'image/object/class/text':
                        category_names,
                    'image/object/class/label':
                        category_ids,
                    'image/object/is_crowd':
                        is_crowd,
                    'image/object/area':
                        area,
                    })

            example = feature_dict

            

            # return True, filename, example
        json_file.close()

testing()