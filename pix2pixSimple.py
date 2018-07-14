from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time


EPS = 1e-12
CROP_SIZE = 256
scale_size = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image



def load_examples(input_dir):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        print(contents.shape)
        raw_input = decode(contents)
        print(contents.shape)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        #a_images = preprocess(raw_input[:,:width//2,:])
        #b_images = preprocess(raw_input[:,width//2:,:])
        a_images = preprocess(raw_input[:,:,:])
        b_images = preprocess(raw_input[:,:,:])
        

    inputs, targets = [b_images, a_images]
    

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=1)
    steps_per_epoch = int(math.ceil(len(input_paths)))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    ngf = 64
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs):
    

    with tf.variable_scope("generator"):
        out_channels = int(inputs.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    return Model(
        predict_real=None, #predict_real,
        predict_fake=None, #predict_fake,
        discrim_loss=None, #ema.average(discrim_loss)
        discrim_grads_and_vars=None, #discrim_grads_and_vars,
        gen_loss_GAN=None, #ema.average(gen_loss_GAN),
        gen_loss_L1=None, #ema.average(gen_loss_L1),
        gen_grads_and_vars=None, #gen_grads_and_vars,
        outputs=outputs,
        train=None, #tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    name = "outputIm"
    for kind in ["inputs", "outputs"]: #, "targets"
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        #fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
    #filesets.append(fileset)
    #return filesets


def main(checkpoint, input_dir, output_dir):
    seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    #options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    
    # disable these features in test mode
    scale_size = CROP_SIZE


    #with open(os.path.join(output_dir, "options.json"), "w") as f:
    #    f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    
    examples = load_examples(input_dir)
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs) #, examples.targets
    
    inputs = deprocess(examples.inputs)
    #targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    #with tf.name_scope("convert_targets"):
    #    converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            #"targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    saver = tf.train.Saver(max_to_keep=1)

    logdir = output_dir #if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:

        if checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(checkpoint)
            saver.restore(sess, checkpoint)

        # testing
        # at most, process the test data once
        start = time.time()
        results = sess.run(display_fetches)
        filesets = save_images(results, output_dir)
        #for i, f in enumerate(filesets):
        #    print("evaluated image", f["name"])
        print("Finished writing")

def convert(image):
    #if aspect_ratio != 1.0:
    #    # upscale to correct aspect ratio
    size = [CROP_SIZE, CROP_SIZE ]
    image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    
def init(checkpoint, output_dir):
    seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    raw_input = tf.placeholder(tf.float32, [1,None, None, 3])
    
    with tf.name_scope("load_images"):
        #path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
        #reader = tf.WholeFileReader()
        #paths, contents = reader.read(path_queue)
        #raw_input = decode(contents)
        #raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        #assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        #with tf.control_dependencies([assertion]):
        #    raw_input = tf.identity(raw_input)

        #raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        inputs = preprocess(raw_input)
        

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(inputs) #, examples.targets
    
    inputs = deprocess(inputs)
    outputs = deprocess(model.outputs)
    
    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)


    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        
    
    saver = tf.train.Saver(max_to_keep=1)

    #logdir = output_dir #if (a.trace_freq > 0 or a.summary_freq > 0) else None
    #sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    
    sess = tf.Session()
    if checkpoint is None:
        raise Exception("checkpoint required for test mode")
    else:
        print("loading model from checkpoint")
        
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver.restore(sess, checkpoint)
    return sess, raw_input, display_fetches
        

import cv2
from PIL import Image
def generate(sess, raw_input, display_fetches, input_path, output_dir):
    

    #input_im = Image.open(input_path).convert('RGB')
    #input_im = input_im.resize((256,256))
    #print(input_im.size)
    input_im = cv2.imread(input_path)
    input_im = cv2.resize(input_im, (256,256))
    #input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
    print(input_im.shape)
    # testing
    # at most, process the test data once
    start = time.time()
    results = sess.run(display_fetches, feed_dict={raw_input: [input_im]})
    filesets = save_images(results, output_dir)
    #for i, f in enumerate(filesets):
    #    print("evaluated image", f["name"])
    print("Finished writing")


#checkpoint="edges_train/"
checkpoint="old_checkpoint/"
input_dir="images/" #bmw2.jpg
output_dir="results"

#sess, raw_input, display_fetches = init(checkpoint, output_dir)
#generate(sess, raw_input, display_fetches, input_dir, output_dir)
#sess.close()
main(checkpoint, input_dir, output_dir)

