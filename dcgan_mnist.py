from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from gan import GAN
from util import get_log_dir, leaky_relu

class DCGAN(GAN):
    """
    DCGAN on MNIST
    """
    def _create_generator(self):
        batch_normalization = tf.layers.batch_normalization # Could not import ...
        conv2d_transpose = tf.layers.conv2d_transpose
        dense = tf.layers.dense

        z = dense(self._z, units=64*7*7, activation=leaky_relu, name='fc_1')
        z = tf.reshape(z,[-1,7,7,64],name='flatten')
        z = batch_normalization(z, center=True, scale=True, training=True,name='bn_1')
        z = conv2d_transpose(z, filters=32, kernel_size=[5,5], strides=(2,2), padding='same', activation=leaky_relu, name='deconv_2')
        z = batch_normalization(z, center=True, scale=True, training=True,name='bn_2')
        z = conv2d_transpose(z, filters=1, kernel_size=[5,5], strides=(2,2), padding='same', activation=None, name='deconv_1')
        x_gan = tf.nn.sigmoid(z,'x_gan')
        return x_gan

    def _create_discriminator(self,x_in):
        """
        Create the discriminator network. The GAN base Class will call this method twice. Once for the
        :param x_in: the tensor that is the input to the discriminator
        :return:
        """
        x = tf.layers.conv2d(x_in, filters=32, kernel_size=[5,5], strides=(2,2), padding='same', activation=leaky_relu, name='conv_1')
        x = tf.layers.batch_normalization(x, center=True, scale=True, training=True,name='bn_1')

        x = tf.layers.conv2d(x, filters=64, kernel_size=[3,3], strides=(2,2), padding='same', activation=leaky_relu, name='conv_2')
        x = tf.layers.batch_normalization(x, center=True, scale=True, training=True,name='bn_2')

        x = tf.reshape(x,[-1,7*7*64],name='flatten')
        x = tf.layers.dense(x,units=1,activation=None,name='fc1') # sigmoid is applied implicitly in the computation of the loss
        return x

if __name__ == '__main__':
    _BATCH_SIZE = 128 # Size of the minibatches
    _CODE_SIZE = 10 # Size of the latent code
    _MAX_IT = 20000 # Number of iterations

    _LOG_BASE_DIR = 'log'
    _LOG_DIR_SUFFIX = 'mnist_gan'
    _LOG_DIR = get_log_dir(_LOG_BASE_DIR, _LOG_DIR_SUFFIX)

    _CHECKPOINT_FILE ='%s/model.ckpt'%_LOG_DIR

    # Load mnist data and create a function to get the batches
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    mnist_batch = lambda x: mnist.train.next_batch(x)[0].reshape(x,28,28,1)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Create the network and initialise the parameters
        gan = DCGAN([28,28,1],_BATCH_SIZE,_CODE_SIZE)
        sess.run(tf.global_variables_initializer())
        # Create file writers and summary
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(_LOG_DIR,sess.graph)

        for i in range(_MAX_IT):
            _,_,summary = gan.perform_training_step(mnist_batch(_BATCH_SIZE),summary=[merged])
            train_writer.add_summary(summary,i)
            if i % 1000 == 0 or i == _MAX_IT-1:
                saver.save(sess, _CHECKPOINT_FILE)
                print("Model saved at iteration %d"%i)
