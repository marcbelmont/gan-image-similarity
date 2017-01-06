import numpy as np
import os.path
from hashlib import md5
from glob import glob
import pickle
import tensorflow as tf

CACHE_PATH = '/tmp/tf-cache'


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image
    return img


def count_params(variables, scopes):
    results = []
    for pattern in scopes:
        count = sum([reduce(lambda a, b: a * b, v.get_shape().as_list(), 1)
                     for v in variables if pattern in v.name])
        results += ['%s: %.3fM' % (pattern, count / 1e6)]
    print(', '.join(results))


def cache_result(func):
    def func_wrapper(*args):
        # Get unique code
        checkpoint = tf.train.latest_checkpoint(args[0].logdir)
        m = md5()
        m.update(checkpoint)
        m.update(args[0].file_pattern)
        code = m.hexdigest()

        # Load or calculate result
        cdata = cache_load(code)
        if cdata:
            return cdata
        result = func(*args)
        cache_save(code, *result)
        return result
    return func_wrapper


def cache_save(code, *args):
    print('Saving results')
    tf.gfile.MakeDirs(CACHE_PATH)
    result = []
    for i, x in enumerate(args):
        if type(x) == list:
            with open(CACHE_PATH + '/%s-%s.pkl' % (i, code), 'wb') as f:
                result += [pickle.dump(x, f)]
        else:
            result += [np.save(CACHE_PATH + '/%s-%s.npy' % (i, code), x)]
    return result


def cache_load(code):
    result = []
    for path in sorted(glob(CACHE_PATH + '/*%s*' % code)):
        _, ext = os.path.splitext(path)
        if ext == '.pkl':
            with open(path, 'rb') as f:
                result += [pickle.load(f)]
        else:
            result += [np.load(path)]
    return result
