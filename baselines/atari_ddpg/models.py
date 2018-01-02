import tensorflow as tf
#import tensorflow.contrib as tc
from baselines.atari_ddpg.utils import conv, fc, conv_to_fc
import numpy as np
import tensorflow.contrib as tc
#from baselines.atari_ddpg.nn import conv2d
class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

class Actor(Model):
     def __init__(self, nb_actions, batch_norm, name='actor', layer_norm=True):
         super(Actor, self).__init__(name=name)
         self.nb_actions = nb_actions
         self.layer_norm = layer_norm
         self.batch_norm = batch_norm


     def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            if self.batch_norm == True:
                h = tc.layers.batch_norm(conv(tf.cast(x, tf.float32)/255., 'c1', act=lambda x:x, nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
                h = tf.nn.relu(h)
                h2 = tc.layers.batch_norm(conv(h, 'c2', nf=64, rf=4, stride=2, act=lambda x:x, init_scale=np.sqrt(2)))
                h2 = tf.nn.relu(h2)
                h3 = tc.layers.batch_norm(conv(h2, 'c3', nf=64, rf=3, stride=1, act=lambda x:x, init_scale=np.sqrt(2)))
                h3 = tf.nn.relu(h3)
                h3 = conv_to_fc(h3)
            else:
                h = conv(tf.cast(x, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
                h3 = conv_to_fc(h3)

            if self.layer_norm == True:
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), act= lambda y: y)
                h4 = tc.layers.layer_norm(h4, center=True, scale=True)
                h4 = tf.nn.relu(h4)
            else:
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

            pi = fc(h4, 'pi', self.nb_actions, act=tf.nn.softmax)
            return pi


class Critic(Model):
    def __init__(self, nb_actions, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm


    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            h = conv(tf.cast(x, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            if self.layer_norm == True:
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), act= lambda y: y)
                h4 = tc.layers.layer_norm(h4, center=True, scale=True)
                h4 = tf.nn.relu(h4)

                h4_action = fc(action, 'fc_action', nh=128, init_scale=np.sqrt(2), act= lambda y: y)
                h4_action = tc.layers.layer_norm(h4_action, center=True, scale=True)
                h4_action = tf.nn.relu(h4_action)

                h5 = fc(tf.concat([h4, h4_action], axis=1), 'h5', nh=512, act= lambda y: y)
                h5 = tc.layers.layer_norm(h5, center=True, scale=True)
                h5 = tf.nn.relu(h5)

                vf = fc(h5, 'v', 1, act=lambda x:x)
                return vf


            else:
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
                h4_action = fc(action, 'fc_action', nh=128, init_scale=np.sqrt(2))
                h5 = fc(tf.concat([h4, h4_action], axis=1), 'h5', 512, act=tf.nn.relu)
                vf = fc(h5, 'v', 1, act=lambda x:x)

                return vf


    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
