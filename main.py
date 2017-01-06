from __future__ import print_function
from glob import glob
from helpers import merge, count_params, cache_result
from random import randint
from zap50k import zap_data, IMAGE_SIZE
import itertools
import json
import math
import numpy as np
import os
import scipy.misc
import tensorflow as tf
import time

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

TINY = 1e-8

#########
# Flags #
#########

flags = tf.app.flags
flags.DEFINE_string("file_pattern", "ut-zap50k-images/*/*/*/*.jpg", "Pattern to find zap50k images")
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir", None, "Directory to save samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_boolean("similarity", False, "Find most similar shoe")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS

if FLAGS.debug:
    tf.set_random_seed(1)
    np.random.seed(1)

##################
# Model settings #
##################

Z_DIM = 80
C_DIM = 8
C_COEFF = .05

##########
# Models #
##########


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator(z, latent_c):
    depths = [32, 64, 64, 64, 64, 64, 3]
    sizes = zip(
        np.linspace(4, IMAGE_SIZE['resized'][0], len(depths)).astype(np.int),
        np.linspace(6, IMAGE_SIZE['resized'][1], len(depths)).astype(np.int))
    with slim.arg_scope([slim.conv2d_transpose],
                        normalizer_fn=slim.batch_norm,
                        kernel_size=3):
        with tf.variable_scope("gen"):
            size = sizes.pop(0)
            net = tf.concat(1, [z, latent_c])
            net = slim.fully_connected(net, depths[0] * size[0] * size[1])
            net = tf.reshape(net, [-1, size[0], size[1], depths[0]])
            for depth in depths[1:-1] + [None]:
                net = tf.image.resize_images(
                    net, sizes.pop(0),
                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                if depth:
                    net = slim.conv2d_transpose(net, depth)
            net = slim.conv2d_transpose(
                net, depths[-1], activation_fn=tf.nn.tanh, stride=1, normalizer_fn=None)
            tf.image_summary("gen", net, max_images=8)
    return net


def discriminator(input, reuse, dropout, int_feats=False, c_dim=None):
    depths = [16 * 2**x for x in range(5)] + [16]
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        reuse=reuse,
                        normalizer_fn=slim.batch_norm,
                        activation_fn=lrelu):
        with tf.variable_scope("discr"):
            net = input
            for i, depth in enumerate(depths):
                if i != 0:
                    net = slim.dropout(net, dropout, scope='dropout')
                if i % 2 == 0:
                    net = slim.conv2d(
                        net, depth, kernel_size=3, stride=2, scope='conv%d' % i)
                else:
                    net = slim.conv2d(
                        net, depth, kernel_size=3, scope='conv%d' % i)
            net = slim.flatten(net)
            if int_feats:
                return net
            else:
                d_net = slim.fully_connected(
                    net, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope='out')
    if c_dim:
        with tf.variable_scope('latent_c'):
            q_net = slim.fully_connected(
                net, c_dim, activation_fn=tf.nn.tanh, scope='out')
        return d_net, q_net
    return d_net


def loss(d_model, g_model, dg_model, q_model, latent_c):
    t_vars = tf.trainable_variables()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Latent_C
    q_loss = tf.reduce_sum(0.5 * tf.square(latent_c - q_model)) * C_COEFF

    # Discriminator
    d_loss = -tf.reduce_mean(tf.log(d_model + TINY) + tf.log(1. - dg_model + TINY))
    tf.scalar_summary('d_loss', d_loss)
    d_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
        d_loss + q_loss,
        global_step=global_step,
        var_list=[v for v in t_vars if 'discr/' in v.name or 'latent_c/' in v.name])

    # Generator
    g_loss = -tf.reduce_mean(tf.log(dg_model + TINY))
    tf.scalar_summary('g_loss', g_loss)
    g_trainer = tf.train.AdamOptimizer(.001, beta1=.5).minimize(
        g_loss + q_loss,
        var_list=[v for v in t_vars if 'gen/' in v.name or 'latent_c/' in v.name])

    return d_trainer, d_loss, g_trainer, g_loss, global_step


#######
# GAN #
#######


def gan(dataset, sess):
    # Model
    x = tf.placeholder(tf.float32, shape=[
        None, IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3])
    dropout = tf.placeholder(tf.float32)
    d_model = discriminator(x, reuse=False, dropout=dropout)

    z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
    latent_c = tf.placeholder(shape=[None, C_DIM], dtype=tf.float32)
    g_model = generator(z, latent_c)
    dg_model, q_model = discriminator(
        g_model, reuse=True, dropout=dropout, c_dim=C_DIM)

    d_trainer, d_loss, g_trainer, g_loss, global_step = loss(
        d_model, g_model, dg_model, q_model, latent_c)

    # Stats
    t_vars = tf.trainable_variables()
    count_params(t_vars, ['discr/', 'gen/', 'latent_c/'])
    # for v in t_vars:
    # tf.histogram_summary(v.name, v)

    # Init
    summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
    tf.initialize_all_variables().run()

    # Saver
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    if checkpoint:
        print('Restoring from', checkpoint)
        saver.restore(sess, checkpoint)

    # Dataset queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.train.start_queue_runners(sess=sess)

    # Training loop
    for step in range(global_step.eval(), 1 if FLAGS.debug else int(1e6)):
        z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, Z_DIM]).astype(np.float32)
        c_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, C_DIM])
        images, _ = sess.run(dataset['batch'])
        feed_dict = {z: z_batch, latent_c: c_batch, x: images, dropout: .5, }

        # Update discriminator
        start = time.time()
        _, d_loss_val = sess.run([d_trainer, d_loss], feed_dict=feed_dict)
        d_time = time.time() - start

        # Update generator
        start = time.time()
        _, g_loss_val, summary_str = sess.run([g_trainer, g_loss, summary], feed_dict=feed_dict)
        g_time = time.time() - start

        # Log details
        if step % 10 == 0 or FLAGS.debug:
            print("[%s, %s] Disc loss: %.3f (%.2fs), Gen Loss: %.3f (%.2fs)" %
                  (step, step * FLAGS.batch_size / dataset['size'], d_loss_val, d_time, g_loss_val, g_time, ))
            summary_writer.add_summary(summary_str, global_step.eval())

        # Early stopping
        if np.isnan(g_loss_val) or np.isnan(d_loss_val):
            print('Early stopping', g_loss_val, d_loss_val)
            break

        # save model
        if step % 1000 == 0 and not FLAGS.debug:
            print('Saving')
            checkpoint_file = os.path.join(FLAGS.logdir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=global_step)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    return


##########
# Sample #
##########

def sample(FLAGS, sess):
    # Model
    z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
    latent_c = tf.placeholder(shape=[None, C_DIM], dtype=tf.float32)
    g_model = generator(z, latent_c)

    # Restore
    saver = tf.train.Saver()
    checkpoints = [x for x in glob(FLAGS.logdir + '/checkpoint-*') if 'meta' not in x]
    checkpoints = [tf.train.latest_checkpoint(FLAGS.logdir)]
    for checkpoint in checkpoints:
        saver.restore(sess, checkpoint)

        # Save samples
        output = "samples/%s.png" % os.path.basename(checkpoint)
        samples = 144
        width = math.sqrt(samples)

        # Input
        z_batch = np.random.uniform(-1.0, 1.0, size=[samples, Z_DIM]).astype(np.float32)
        c_batch = np.zeros((samples, C_DIM))
        if 0:
            for i in range(8):
                c_batch[i * width:(i + 1) * width, i] = np.linspace(-1, 1, width)
        else:
            c_batch[:, 0] = np.tile(np.linspace(-1, 1, width), width)
            c_batch[:, 1] = np.repeat(np.linspace(-1, 1, width), width)

        # Run and save
        images = sess.run(g_model, feed_dict={z: z_batch, latent_c: c_batch})
        images = np.reshape(
            images, [samples, IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3])
        images = (images + 1.) / 2.
        scipy.misc.imsave(output, merge(images, [int(width)] * 2))


##############
# Similarity #
##############

@cache_result
def export_intermediate(FLAGS, sess, dataset):
    # Models
    x = tf.placeholder(tf.float32, shape=[
        None, IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3])
    dropout = tf.placeholder(tf.float32)
    feat_model = discriminator(x, reuse=False, dropout=dropout, int_feats=True)

    # Init
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Restore
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    saver.restore(sess, checkpoint)

    # Run
    all_features = np.zeros((dataset['size'], feat_model.get_shape()[1]))
    all_paths = []
    for i in itertools.count():
        try:
            images, paths = sess.run(dataset['batch'])
        except tf.errors.OutOfRangeError:
            break
        if i % 10 == 0:
            print(i * FLAGS.batch_size, dataset['size'])
        im_features = sess.run(feat_model, feed_dict={x: images, dropout: 1, })
        all_features[FLAGS.batch_size * i:FLAGS.batch_size * i + im_features.shape[0]] = im_features
        all_paths += list(paths)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    return all_features, all_paths


def similarity(FLAGS, sess, all_features, all_paths):
    def select_images(distances):
        indices = np.argsort(distances)
        images = []
        size = 40
        for i in range(size):
            images += [dict(path=all_paths[indices[i]],
                            index=indices[i],
                            distance=distances[indices[i]])]
        return images

    # Distance
    x1 = tf.placeholder(tf.float32, shape=[None, all_features.shape[1]])
    x2 = tf.placeholder(tf.float32, shape=[None, all_features.shape[1]])
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x1, x2)), reduction_indices=1))

    # Init
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess.run(init_op)

    #
    clip = 1e-3
    np.clip(all_features, -clip, clip, all_features)

    # Get distances
    result = []
    bs = 100
    needles = [randint(0, all_features.shape[0]) for x in range(10)]
    for needle in needles:
        item_block = np.reshape(np.tile(all_features[needle], bs), [bs, -1])
        distances = np.zeros(all_features.shape[0])
        for i in range(0, all_features.shape[0], bs):
            if i + bs > all_features.shape[0]:
                bs = all_features.shape[0] - i
            distances[i:i + bs] = sess.run(
                l2diff, feed_dict={x1: item_block[:bs], x2: all_features[i:i + bs]})

        # Pick best matches
        result += [select_images(distances)]

    with open('logs/data.json', 'w') as f:
        json.dump(dict(data=result), f)
    return


########
# Main #
########

def main(_):
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)
    if FLAGS.sampledir and not tf.gfile.Exists(FLAGS.sampledir):
        tf.gfile.MakeDirs(FLAGS.sampledir)

    with tf.Session() as sess:
        if FLAGS.sampledir:
            sample(FLAGS, sess)
        elif FLAGS.similarity:
            dataset = zap_data(FLAGS, False)
            all_features, all_paths = export_intermediate(FLAGS, sess, dataset)
            similarity(FLAGS, sess, all_features, all_paths)
        else:
            dataset = zap_data(FLAGS, True)
            gan(dataset, sess)


if __name__ == '__main__':
    tf.app.run()
