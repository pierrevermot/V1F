#import tensorflow as tf
import numpy as np    
import glob
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from joblib import Parallel, delayed

def npy_to_tfrecords_old(inputs, outputs, filename):
    # @Vijay Mariappan on stackoverflow
  with tf.io.TFRecordWriter(filename) as writer:
    for X, y in zip(inputs, outputs):
        # Feature contains a map of string to feature proto objects
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()

        # write the serialized objec to the disk
        writer.write(serialized)

  
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array
      
        
def npy_to_tfrecords(inputs, outputs, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for X, y in zip(inputs, outputs):
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            serialized_X_array = serialize_array(X)
            serialized_y_array = serialize_array(y)
            feature = {'X': _bytes_feature(serialized_X_array), 'y': _bytes_feature(serialized_y_array)}
            example_message = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_message.SerializeToString())


# example proto decode
def _parse_function(example_proto):
    keys_to_features = {'X': tf.io.FixedLenFeature(shape=(128, 128, 32), dtype=tf.float32),
                      'y': tf.io.FixedLenFeature(shape=(128, 128), dtype=tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return parsed_features['X'], parsed_features['y']


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


#%%
name_ims = "fpm_128"
name_obs = "NACO_M"
rep_bulk_ims = os.environ.get('SCRATCH')+'/'+name_ims
rep_bulk_obs = os.environ.get('SCRATCH')+'/'+name_ims+'_'+name_obs

print(rep_bulk_ims)
print(rep_bulk_obs)

filenames_ims = glob.glob(rep_bulk_ims+'/IMS*')


def process(filenames_ims):
    k = 0
    for filename_ims in filenames_ims:
        filename_obs = filename_ims.replace(rep_bulk_ims, rep_bulk_obs).replace("/IMS_", "/OBS_")
        npy_array_ims = np.load(filename_ims)
        npy_array_ims = np.array(npy_array_ims, dtype=np.single)
        npy_array_obs = np.load(filename_obs)
        npy_array_obs = np.array(npy_array_obs, dtype=np.single)
        npy_array_obs = np.moveaxis(npy_array_obs, 1, 3)
        npy_to_tfrecords(npy_array_obs, npy_array_ims, filename_obs.replace("OBS_", "").replace(".npy", ".tfrecords"))
        k += 1

import os
n_cores = os.cpu_count()

filenames = split(filenames_ims, n_cores)
Parallel(n_jobs=n_cores, backend="threading")(delayed(process)(fn) for fn in filenames)

