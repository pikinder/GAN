"""
Basic Generative Adversial Network (GAN) Implementation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class GAN(object):
    def __init__(self, x_shape, batch_size, code_size):
        """
        :param x_shape:
        :param batch_size:
        :param code_size:
        :return:
        """
        self.x_shape = x_shape
        self.x_batch_shape = [batch_size]
        self.x_batch_shape.extend(x_shape)
        self.batch_size = batch_size
        self.code_size = code_size
        self._LABEL_DATA = .9
        self._LABEL_GAN = .0

        with tf.name_scope('inputs'):
            self._z = tf.placeholder(tf.float32, [self.batch_size, self.code_size], name='z_input')
            self._x_data = tf.placeholder(tf.float32, self.x_batch_shape, name='x_input')

        with tf.variable_scope('generator'):
            self._x_gan = self._create_generator()

        with tf.variable_scope('discriminator') as scope:
            self._y_gan = self._create_discriminator(self._x_gan)
            scope.reuse_variables()
            self._y_data = self._create_discriminator(self._x_data)

        with tf.name_scope('optimisation'):
            self._loss_disc, self._loss_gen = self._create_losses()
            self._train_disc, self._train_gen = self._create_training_operations()

        self._create_summaries()

    def _create_summaries(self):
        """

        :return:
        """
        tf.summary.scalar(name='ce_discriminator', tensor=tf.reduce_mean(self._loss_disc))
        tf.summary.scalar(name='ce_generator', tensor=tf.reduce_mean(self._loss_gen))

        tf.summary.histogram(name='p_data', values=tf.nn.sigmoid(self._y_data))
        tf.summary.histogram(name='p_gan', values=tf.nn.sigmoid(self._y_gan))

        tf.summary.image(name='x_data', tensor=self._x_data, max_outputs=10)
        tf.summary.image(name='x_gan', tensor=self._x_gan, max_outputs=10)

    def _get_session(self, session):
        """
        Helper method to facilitate the use of a default tensorflow session
        :param session:
        :return:
        """
        if session is None:
            return tf.get_default_session()

    def _create_generator(self):
        """

        :return:
        """
        raise NotImplementedError('Subclass resp')

    def _create_discriminator(self, x):
        """

        :param x:
        :return:
        """
        raise NotImplementedError('Subclass resp')

    def _create_losses(self):
        """

        :return:
        """
        ce = tf.nn.sigmoid_cross_entropy_with_logits
        ones = np.ones((self.batch_size, 1)).astype(np.float32)

        loss_disc_data = ce(logits=self._y_data, labels=tf.constant(self._LABEL_DATA * ones), name='ce_disc_data')
        loss_disc_gan = ce(logits=self._y_gan, labels=tf.constant(self._LABEL_GAN * ones), name='ce_disc_gan')
        loss_gen = ce(logits=self._y_gan, labels=tf.constant(self._LABEL_DATA * ones), name='ce_gan')

        return tf.add(loss_disc_data, loss_disc_gan,'ce_disc'), loss_gen

    def _create_training_operations(self):
        """

        :return:
        """
        variables = tf.trainable_variables()
        generator_variables = [v for v in variables if v.name.startswith('generator/')]
        discriminator_variables = [v for v in variables if v.name.startswith('discriminator/')]

        # Train on the true data, only discriminator
        train_disc = tf.train.AdamOptimizer().minimize(self._loss_disc, var_list=discriminator_variables)
        train_gen = tf.train.AdamOptimizer().minimize(self._loss_gen, var_list=generator_variables)

        return train_disc, train_gen

    def sample_code(self):
        """
        Sample the gan input code
        :return:
        """
        return np.random.randn(self.batch_size, self.code_size)

    def generate_sample(self, session=None):
        """
        sample from the gan
        :param session:
        :return:
        """
        session = self._get_session(session)
        return session.run(self._x_gan, feed_dict={self._z: self.sample_code()})

    def perform_training_step(self, x, summary=None, session=None):
        """

        :param x:
        :param summary:
        :param session:
        :return:
        """
        session = self._get_session(session)
        feed_dict = {
            self._x_data: x,
            self._z: self.sample_code()
        }
        session_operations = [self._train_disc,
                              self._train_gen
                              ]
        if summary is not None:
            session_operations.extend(summary)
        return session.run(session_operations, feed_dict=feed_dict)
