import sys
sys.path.append('./networks/unet/')

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, LeakyReLU
from tensorflow.keras.models import Model
import glob
import config as pm  # migrated from params_model
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx('float32')

def get_activation(string):
    if string == 'LeakyReLU':
        return LeakyReLU()
    else:
        return Activation(string)

def get_model():
    inputs = Input((pm.n_pix, pm.n_pix, pm.n_frames_in))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = get_activation(pm.train.inner_activation)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = get_activation(pm.train.inner_activation)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = get_activation(pm.train.inner_activation)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = get_activation(pm.train.inner_activation)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = get_activation(pm.train.inner_activation)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = get_activation(pm.train.inner_activation)(conv3)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), padding='same')(up4)
    conv4 = get_activation(pm.train.inner_activation)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
    conv4 = get_activation(pm.train.inner_activation)(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), padding='same')(up5)
    conv5 = get_activation(pm.train.inner_activation)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
    conv5 = get_activation(pm.train.inner_activation)(conv5)

    output = Conv2D(pm.n_frames_out, (1, 1))(conv5)
    output = get_activation(pm.train.output_activation)(output)
    output_reshaped = Reshape((pm.n_pix, pm.n_pix, pm.n_frames_out))(output)
    return Model(inputs=inputs, outputs=output_reshaped)

def load_xy(file_path):
    # Extracting the file prefix to pair x and y
    num = tf.strings.regex_replace(file_path, ".npy", "")
    num = tf.strings.regex_replace(num, "*_", "")
    
    # Creating paths for x and y files
    x_file_path = tf.strings.join(["OBS_", num, ".npy"])
    y_file_path = tf.strings.join(["IMS_", num, ".npy"])
    
    # Reading the content of the files
    x_content = tf.io.read_file(x_file_path)
    y_content = tf.io.read_file(y_file_path)
    
    # You might want to do additional processing here like converting strings to numbers,
    # reshaping, etc., based on the format of your data
    
    return x_content, y_content

# example proto decode
def _parse_function(example_proto):
    keys_to_features = {'X': tf.io.FixedLenFeature(shape=(128, 128, 32), dtype=tf.float32),
                      'y': tf.io.FixedLenFeature(shape=(128, 128), dtype=tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return parsed_features['X'], parsed_features['y']

def parse_function(element):
  parse_dic = {
    'X': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32
    'y': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32

    }
  example_message = tf.io.parse_single_example(element, parse_dic)

  X_feature = example_message['X'] # get byte string
  X = tf.io.parse_tensor(X_feature, out_type=tf.float32) # restore 2D array from byte string
  y_feature = example_message['y'] # get byte string
  y = tf.io.parse_tensor(y_feature, out_type=tf.float32) # restore 2D array from byte string

  return X, y

import os
name = "decK_UNET"
rep_bulk = os.environ.get('SCRATCH')+'/'+name

import idr_tf # IDRIS package available in all TensorFlow modules

def train():
    # filenames = glob.glob('/home/pierre/Documents/ASTRONEURON/V1/temp/decK_UNET/*.tfrecords')
    # dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.interleave(tf.data.TFRecordDataset,
    #                              num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    
    # dataset = dataset.map(_parse_function, 
    #                       num_parallel_calls=tf.data.AUTOTUNE, 
    #                       deterministic=False)

    filenames = glob.glob(rep_bulk+'/*.tfrecords')
    #dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.interleave(tf.data.TFRecordDataset,
    #                               num_parallel_calls=2, deterministic=False)
    # dataset = dataset.interleave(tf.data.TFRecordDataset)
    
    # dataset = dataset.map(_parse_function, 
    #                       num_parallel_calls=tf.data.AUTOTUNE, 
    #                       deterministic=False)
    #dataset = dataset.map(parse_function)
    num_parallel_calls = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.list_files(rep_bulk+'/*.tfrecords')
    num_workers = idr_tf.size
    worker_index = idr_tf.rank
    print(num_workers, worker_index)
    dataset = dataset.shard(num_workers,worker_index)
    mod = get_model()
    opt = Adam(learning_rate=5e-5, clipnorm=5)
    loss = tf.keras.losses.MeanAbsoluteError()
    mod.compile(loss=loss, optimizer=opt)
    batch_size = 64
    prefetch_factor = tf.data.experimental.AUTOTUNE
    print(prefetch_factor, num_parallel_calls)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                 cycle_length=num_parallel_calls, block_length=1)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(prefetch_factor)
#    dataset = dataset.repeat(num_epochs)
#    dataset = dataset.batch(batch_size, drop_remainder=True)#.prefetch(1)
    mod.fit(dataset, epochs=50)

train()
"""
height, width, depth = 128, 128, 32

def numpy_generator(x_filenames):
    for x_file, y_file in x_filenames:
        y_file = x_file.replace("OBS_", "IMS_")
        X = np.load(x_file)
        Y = np.load(y_file)
        yield X, Y

import gc

def numpy_generator(x_filenames):
    for x_file in x_filenames:
        y_file = x_file.replace("OBS_", "IMS_")
        X = np.load(x_file)
        X = np.moveaxis(X, 1, 3)
        Y = np.load(y_file)
        for x, y in zip(X, Y):
            yield x, y
        del X
        del Y
        gc.collect()

def create_dataset(x_filenames, batch_size):
    # Create a dataset from a generator
    dataset = tf.data.Dataset.from_generator(
        #numpy_generator,
        lambda: numpy_generator(x_filenames),
        output_signature=(tf.TensorSpec(shape=(height, width, depth), dtype=tf.float32),  # Adjust the shape and dtype
                          tf.TensorSpec(shape=(height, width), dtype=tf.float32)),  # Adjust the shape and dtype
        args=(x_filenames,)
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=10000)  # Adjust buffer size as needed
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

def create_dataset(x_filenames, batch_size):
    # Create a dataset from a generator
    dataset = tf.data.Dataset.from_generator(
        lambda: numpy_generator(x_filenames),  # Using a lambda to pass arguments to the generator
        output_signature=(tf.TensorSpec(shape=(height, width, depth), dtype=tf.float32),  # Adjust the shape and dtype
                          tf.TensorSpec(shape=(height, width), dtype=tf.float32))  # Adjust the shape and dtype
    )
    
    dataset = dataset.batch(batch_size)
#    dataset = dataset.shuffle(buffer_size=10000)  # Adjust buffer size as needed
#    dataset = dataset.prefetch(buffer_size=1)#tf.data.experimental.AUTOTUNE)
    
    return dataset


import os
name = "decK_UNET"
rep_bulk = os.environ.get('SCRATCH')+'/'+name


def train():
    # Usage
    x_filenames = glob.glob(rep_bulk+'/OBS_*.npy')
    batch_size = 8  # Adjust batch size as needed
    dataset = create_dataset(x_filenames, batch_size)
    mod = get_model()
    opt = Adam(learning_rate=5e-5, clipnorm=5)
    loss = tf.keras.losses.MeanAbsoluteError()
    mod.compile(loss=loss, optimizer=opt)
    batch_size = 32
    prefetch_factor = tf.data.experimental.AUTOTUNE
    print(prefetch_factor)
#    dataset = dataset.batch(batch_size, drop_remainder=True)#.prefetch(prefetch_factor)
    mod.fit(dataset, epochs=50)


"""
#train()
