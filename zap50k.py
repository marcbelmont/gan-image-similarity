from glob import glob
import tensorflow as tf

IMAGE_SIZE = dict(source=(102, 136, 3), cropped=(100, 132, 3), resized=(64, 96))


def read_image(filename_queue, shuffle):
    image_reader = tf.WholeFileReader()
    path, image_file = image_reader.read(filename_queue)

    # Preprocessing
    image = tf.image.decode_jpeg(image_file, 3)
    if shuffle:
        # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        if image.get_shape()[0] > IMAGE_SIZE['cropped'][0] and image.get_shape()[1] > IMAGE_SIZE['cropped'][1]:
            image = tf.random_crop(image, IMAGE_SIZE['cropped'])
        # image = tf.image.per_image_whitening(image)
    image = tf.image.resize_images(image, IMAGE_SIZE['resized'])
    image = image * (1. / 255) - 0.5
    return [image, path]


def zap_data(FLAGS, shuffle):
    files = glob(FLAGS.file_pattern)
    filename_queue = tf.train.string_input_producer(
        files,
        shuffle=shuffle,
        num_epochs=None if shuffle else 1)
    image = read_image(filename_queue, shuffle)

    # Mini batch
    num_preprocess_threads = 1 if FLAGS.debug else 4
    min_queue_examples = 100 if FLAGS.debug else 10000
    if shuffle:
        images = tf.train.shuffle_batch(
            image,
            batch_size=FLAGS.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * FLAGS.batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images = tf.train.batch(
            image,
            FLAGS.batch_size,
            allow_smaller_final_batch=True)
    # tf.image_summary('images', images, max_images=8)
    return dict(batch=images, size=len(files))
