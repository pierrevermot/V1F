
import os
import tensorflow as tf


# build multi-worker environment from Slurm variables
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)           
 
# use NCCL communication protocol
implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation) 
 
# declare distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver, communication_options=communication_options) 

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, LeakyReLU
from tensorflow.keras.models import Model
import config as pm  # alias for backwards attribute names

from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

if pm.train.float_precision == "mixed":
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
else:
    tf.keras.backend.set_floatx(pm.train.float_precision)

import idr_tf # IDRIS package available in all TensorFlow modules

import time

for rep in [pm.output_dir, pm.datedir, pm.savedir, pm.logdir, pm.plotdir]:
    time.sleep(np.random.random()*10)
    if not os.path.isdir(rep):
        os.mkdir(rep)
    else:
        print('Warning: '+rep+' already exists, check that they are empty')

def get_activation(string):
    if string == 'LeakyReLU':
        return LeakyReLU()
    else:
        return Activation(string)

def get_model():
#    inputs = Input((pm.n_pix, pm.n_pix, pm.n_frames_in))
    # Input layer with flexible dimensions
    inputs = Input(shape=(None, None, pm.n_frames_in))
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
#    output_reshaped = Reshape((pm.n_pix, pm.n_pix, pm.n_frames_out))(output)
    return Model(inputs=inputs, outputs=output)


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
  rf = 2*tf.random.uniform(shape=[])
  return rf*X, rf*y

class PlotValidationPredictionCallback(Callback):
    def __init__(self, validation_data, dirplot):
        super(PlotValidationPredictionCallback, self).__init__()
        self.dirplot = dirplot
        validation_data = validation_data.take(1)  # take 1 batch
        self.validation_data = [(x.numpy(), y.numpy()) for x, y in validation_data][0]

    def on_epoch_end(self, epoch, logs=None):
        truths = self.validation_data[1][:12]
        obss = self.validation_data[0][:12]
        preds = self.model.predict(obss)
        
        plt.figure(figsize=(30, 9))
        for i in range(9):
            truth = truths[i]
            obs = np.mean(obss[i], -1)
            pred = preds[i]
            vmax = np.max(np.concatenate([truth.flatten(), obs.flatten(), pred.flatten()]))
            plt.subplot(3, 9, i+1)
            plt.imshow(truth, cmap='hot')
            plt.axis('off')
            plt.subplot(3, 9, i+1+9)
            plt.imshow(obs, cmap='hot')
            plt.axis('off')
            plt.subplot(3, 9, i+1+2*9)
            plt.imshow(pred, cmap='hot')
            plt.axis('off')
            
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.01)
        plt.savefig(self.dirplot+'/predictions.png')
        plt.savefig(self.dirplot+'/predictions.pdf')
        plt.close()
        
        plt.figure(figsize=(30, 9))
        for i in range(9):
            truth = truths[i]
            obs = np.mean(obss[i], -1)
            pred = preds[i]
            vmax = np.max(np.concatenate([truth.flatten(), obs.flatten(), pred.flatten()]))
            plt.subplot(3, 9, i+1)
            plt.imshow(truth, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-6))
            plt.axis('off')
            plt.subplot(3, 9, i+1+9)
            plt.imshow(obs, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-6))
            plt.axis('off')
            plt.subplot(3, 9, i+1+2*9)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-6))
            plt.axis('off')

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.01)
        plt.savefig(self.dirplot+'/predictions_log_1e6.png')
        plt.savefig(self.dirplot+'/predictions_log_1e6.pdf')
        plt.close()


        plt.figure(figsize=(30, 9))
        for i in range(9):
            truth = truths[i]
            obs = np.mean(obss[i], -1)
            pred = preds[i]
            vmax = np.max(np.concatenate([truth.flatten(), obs.flatten(), pred.flatten()]))
            plt.subplot(3, 9, i+1)
            plt.imshow(truth, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-4))
            plt.axis('off')
            plt.subplot(3, 9, i+1+9)
            plt.imshow(obs, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-4))
            plt.axis('off')
            plt.subplot(3, 9, i+1+2*9)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-4))
            plt.axis('off')
            
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.01)
        plt.savefig(self.dirplot+'/predictions_log_1e4.png')
        plt.savefig(self.dirplot+'/predictions_log_1e4.pdf')
        plt.close()
        
        plt.figure(figsize=(30, 9))
        for i in range(9):
            truth = truths[i]
            obs = np.mean(obss[i], -1)
            pred = preds[i]
            vmax = np.max(np.concatenate([truth.flatten(), obs.flatten(), pred.flatten()]))
            plt.subplot(3, 9, i+1)
            plt.imshow(truth, cmap='hot', norm=LogNorm(vmax=vmax*1e-1, vmin=vmax*1e-3))
            plt.axis('off')
            plt.subplot(3, 9, i+1+9)
            plt.imshow(obs, cmap='hot', norm=LogNorm(vmax=vmax*1e-1, vmin=vmax*1e-3))
            plt.axis('off')
            plt.subplot(3, 9, i+1+2*9)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax*1e-1, vmin=vmax*1e-3))
            plt.axis('off')
            
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.01)
        plt.savefig(self.dirplot+'/predictions_log_1e2.png')
        plt.savefig(self.dirplot+'/predictions_log_1e2.pdf')
        plt.close()


class PlotObsPredictionCallback(Callback):
    def __init__(self, pathsdata, dirplot, datanames=["data"]):
        super(PlotObsPredictionCallback, self).__init__()
        self.pathsdata = pathsdata
        self.dirplot = dirplot
        self.datanames = datanames

    def on_epoch_end(self, epoch, logs=None):
        for k in range(len(self.pathsdata)):
            pathdata = self.pathsdata[k]
            dataname = self.datanames[k]
            
            data = np.load(pathdata)
            datat = np.expand_dims(data, axis=0)
            pred = self.model.predict(datat)[0]
            vmax = np.max(np.concatenate([data.flatten(), pred.flatten()]))
            
            plt.figure(figsize=(18, 9))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(data, 2), cmap='hot')
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='hot')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(self.dirplot+'/'+dataname+'.png')
            plt.savefig(self.dirplot+'/'+dataname+'.pdf')
            plt.close()

            plt.figure(figsize=(18, 9))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(data, 2), cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-6))
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-6))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e6.png')
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e6.pdf')
            plt.close()

            plt.figure(figsize=(18, 9))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(data, 2), cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-4))
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax, vmin=vmax*1e-4))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e4.png')
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e4.pdf')
            plt.close()
            
            plt.figure(figsize=(18, 9))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(data, 2), cmap='hot', norm=LogNorm(vmax=vmax*1e-1, vmin=vmax*1e-3))
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='hot', norm=LogNorm(vmax=vmax*1e-1, vmin=vmax*1e-3))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e2.png')
            plt.savefig(self.dirplot+'/'+dataname+'_log_1e2.pdf')
            plt.close()
        
        
checkpoint_cb = ModelCheckpoint(
    filepath=pm.savedir+'/model_epoch_{epoch:02d}.h5',  # Saves the model with the epoch number in the filename
    save_freq='epoch',  # 'epoch' means save the model after every epoch
    save_best_only=False,  # This means every epoch's model will be saved, not just the 'best' one
    verbose=1  # Logs out when the model is being saved
)

tensorboard_cb = TensorBoard(log_dir=pm.logdir,    histogram_freq=1,  # Record histograms every epoch
    write_graph=True,  # Write the model graph
    write_images=True,  # Write model weights as images
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=5e-6, min_delta=1e-4)


def train():
    num_parallel_calls = tf.data.AUTOTUNE
    num_workers = idr_tf.size
    worker_index = idr_tf.rank
    prefetch_factor = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.list_files(pm.rep_bulk+'/*.tfrecords')
    dataset = dataset.shard(num_workers,worker_index)
    opt = Adam(learning_rate=pm.train.learning_rate, clipnorm=pm.train.clipnorm)
#    opt = Adadelta(learning_rate=pm.learning_rate, clipnorm=pm.clipnorm) 
    loss = tf.keras.losses.MeanAbsoluteError()
    batch_size = pm.train.batch_size
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                 cycle_length=num_parallel_calls, block_length=1)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(prefetch_factor)
    test_dataset = dataset.take(pm.train.test_dataset_length) 
    train_dataset = dataset.skip(pm.train.test_dataset_length)
    
    plot_val_cb = PlotValidationPredictionCallback(test_dataset, pm.plotdir)
    cbs = [checkpoint_cb, tensorboard_cb, plot_val_cb]
    if pm.train.plot_data:
        plot_data_cb = PlotObsPredictionCallback(pm.train.pathsdata, pm.plotdir, pm.train.datanames)
        cbs = [checkpoint_cb, tensorboard_cb, plot_val_cb, plot_data_cb, reduce_lr]
#        cbs = [checkpoint_cb, tensorboard_cb, plot_val_cb, plot_data_cb]
    with strategy.scope():
        mod = get_model()
        mod.compile(loss=loss, optimizer=opt)
    mod.fit(train_dataset, validation_data=test_dataset, callbacks=cbs, epochs=pm.train.epochs)
    mod.evaluate(test_dataset)


