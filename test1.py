# from text_embedding import CLIP
# category_name_string = ';'.join(['flipflop', 'street sign', 'bracelet',
#       'necklace', 'shorts', 'floral camisole', 'orange shirt',
#       'purple dress', 'yellow tee', 'green umbrella', 'pink striped umbrella', 
#       'transparent umbrella', 'plain pink umbrella', 'blue patterned umbrella',
#       'koala', 'electric box','car', 'pole'])
# category_names = [x.strip() for x in category_name_string.split(';')]
# category_names = ['background'] + category_names
# categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
# category_indices = {cat['id']: cat for cat in categories}
# text_embedding = CLIP()
# text_feature = text_embedding(categories)
# print(text_feature)
import tensorflow.compat.v1 as tf
import configs.params_dict as params_dict
import os 
from configs import vild_config
from projects.vild.modeling.vild_model import ViLDModel
from utils import input_utils
from dataloader import mode_keys
import numpy as np
# import tensorflow_datasets as tfds
BATCH_SIZE=1
# RESNET_DEPTH=50
MODEL_DIR="resnet152_vild"
# EVAL_FILE_PATTERN="[DEST_DIR]/val*"
# VAL_JSON_FILE="[DATA_DIR]/lvis_v1_val.json"
RARE_MASK_PATH="resnet152_vild/lvis_rare_masks.npy"
CLASSIFIER_WEIGHT_PATH="resnet152_vild/clip_synonym_prompt.npy"
# CONFIG_FILE="tpu/models/official/detection/projects/vild/configs/[CONFIG_FILE]"
params_override={ "predict": {"predict_batch_size": BATCH_SIZE}, 
#  "eval": {"eval_batch_size": {BATCH_SIZE}, "val_json_file": {VAL_JSON_FILE}, "eval_file_pattern": {EVAL_FILE_PATTERN} }, 
    "frcnn_head": {"classifier_weight_path": CLASSIFIER_WEIGHT_PATH},
    "frcnn_class_loss":{"rare_mask_path": RARE_MASK_PATH}, 
    "postprocess": {"rare_mask_path": RARE_MASK_PATH}}
# print(os.path.exists(RARE_MASK_PATH))

default_config = vild_config.VILD_CFG
restrictions = vild_config.VILD_RESTRICTIONS
params = params_dict.ParamsDict(default_config, restrictions)
checkpoint_path = 'resnet50_vild'
params = params_dict.override_params_dict(params, os.path.join("configs","vild_resnet.yaml"), is_strict=True)
params = params_dict.override_params_dict(params, params_override, is_strict=True)


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
#     path='mnist.npz'
# )
image_path="3ppl.jpg"
model = ViLDModel(params = params)
image_size = 640
with tf.Graph().as_default():
    # outputs = model.build_outputs(
    #     images, {'image_info': images_info}, mode=mode_keys.PREDICT)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size,image_size))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [image_size, image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(
        images, {'image_info': images_info}, mode=mode_keys.PREDICT)

    outputs['detection_boxes'] = (
        outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs
    # model.save_weights('checkpoints/my_checkpoint')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(os.path.join(checkpoint_path,"model.ckpt-180000.meta"))
        # module_file = tf.train.load_checkpoint(checkpoint_path)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(checkpoint_path,"model.ckpt-180000"))
        # saver.restore(sess, os.path.join(checkpoint_path,"model.ckpt"))
        print("start")
        temp = sess.run(predictions,feed_dict={images: input_arr})
        
        print(temp)
        # model.save_weights('checkpoints/my_checkpoint')

# model.keras.load_weights(os.path.join(checkpoint_path, "model.ckpt"))
