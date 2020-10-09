# Create Ultrasound Segmentation Batch Generator Class

import tensorflow as tf
import tensorflow_addons as tfa

def train_preprocess(image, label):

    ###################################################
    #  Random flip L-R
    ###################################################

    do_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)

    # Remove the background class label to do computations faster here
    # then subtract bone label from tensor of ones later.
    label = tf.expand_dims(label[:, :, 1], axis=-1)
    label = tf.cond(do_flip, lambda: tf.image.flip_left_right(label), lambda: label)

    ###################################################
    #  Random rotation L-R
    ###################################################

    angle = tf.random.uniform([], -10, 10)
    pi_on_180 = 0.017453292519943295
    rads = angle * pi_on_180
    image = tfa.image.rotate(image, rads, 'BILINEAR')
    label = tfa.image.rotate(label, rads, 'BILINEAR')

    ###################################################
    #  Random shift L-R and U-D
    ###################################################

    lower_bound = int(-0.1 * 128)
    upper_bound = int(0.1 * 128)
    shift = tf.random.uniform((2,), lower_bound,  upper_bound)
    image = tfa.image.translate(image, shift, 'BILINEAR')
    label = tf.math.round(tfa.image.translate(label, shift, 'BILINEAR'))

    ###################################################
    #  Rescale (zoom in/out and then crop/pad)
    ###################################################

    # Convert to int_32 to be able to differentiate
    # between zeros that was used for padding and
    # zeros that represent a particular semantic class
    label = tf.cast(label, tf.int32)

    # Get height and width tensors
    input_shape = tf.shape(image)[0:2]

    input_shape_float = tf.cast(input_shape, tf.float32)

    final_scale = tf.random.uniform(shape=[1],
                                    minval=0.8,
                                    maxval=1.1,
                                    dtype=tf.float32)

    scaled_input_shape = tf.cast(tf.round(input_shape_float * final_scale), tf.int32)

    # Resize the image and label,
    resized_img = tf.image.resize(image, scaled_input_shape, method='nearest')
    resized_label = tf.image.resize(label, scaled_input_shape, method='nearest')
    resized_label = tf.cast(resized_label, tf.int32)

    cropped_padded_img = tf.image.resize_with_crop_or_pad(resized_img, input_shape[0], input_shape[1])
    cropped_padded_label = tf.image.resize_with_crop_or_pad(resized_label, input_shape[0], input_shape[1])

    ones = tf.ones_like(label)
    ones -= cropped_padded_label
    cropped_padded_label = tf.concat([ones, cropped_padded_label], axis=-1)
    
    image = tf.cast(tf.clip_by_value(cropped_padded_img, 0.0, 1.0), tf.float64)
    label = tf.cast(cropped_padded_label, tf.float32)
    
    return image, label

def generate_weight_maps(image, label):

    coords = tf.meshgrid(tf.range(tf.shape(label)[0]), tf.range(tf.shape(label)[1]))
    coords = tf.stack(coords, axis=-1)
    # Find coordinates that are positive
    mask = label[:, :, 1] > 0
    coords_pos = tf.boolean_mask(coords, mask)
    # Find every pairwise distance
    vec_d = tf.reshape(coords, [-1, 1, 2]) - coords_pos
    # You may choose a difference precision type here
    dists = tf.norm(tf.dtypes.cast(vec_d, tf.float32), axis=-1)
    # Find minimum distances
    min_dists = tf.reduce_min(dists, axis=-1)
    # Reshape & remove infs/nans
    weight_map = tf.expand_dims(tf.reshape(min_dists, [tf.shape(label)[0], tf.shape(label)[1]]), axis=-1)
    weight_map = tf.where(tf.math.is_nan(weight_map), tf.ones_like(weight_map), weight_map)
    weight_map = tf.where(tf.math.is_inf(weight_map), tf.ones_like(weight_map), weight_map)
    weight_map /= tf.reduce_max(weight_map)
    weight_map = tf.concat([weight_map, tf.expand_dims(label[:, :, 1], axis=-1)], axis=-1)
    label = tf.concat([label, weight_map], axis=-1)
    
    return image, label

def train_preprocess_with_maps(image, label):

    image, label = train_preprocess(image, label)
    image, label = generate_weight_maps(image, label)
    
    return image, label
