import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines.common.tf_util import noisy_dense


def model(img_in, num_actions, scope, noisy=False, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)

        with tf.variable_scope("action_value"):
             if noisy:
                # Apply noisy network on fully connected layers
                # ref: https://arxiv.org/abs/1706.10295
                out = noisy_dense(out, name='noisy_fc1', size=16, activation_fn=tf.nn.relu)
                out = noisy_dense(out, name='noisy_fc2', size=num_actions)
             else:
                out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def bootstrap_model(img_in, num_actions, scope, reuse=False):
    """ As described in https://arxiv.org/abs/1602.04621"""
    with tf.variable_scope(scope, reuse=reuse), tf.device("/gpu:0"):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)

        out_list =[]
        with tf.variable_scope("heads"):
            for _ in range(10):
                scope_net = "action_value_head_" + str(_)
                with tf.variable_scope(scope_net):
                    out_temp = out
                    out_temp = layers.fully_connected(out_temp, num_outputs=16, activation_fn=tf.nn.relu)
                    out_temp = layers.fully_connected(out_temp, num_outputs=num_actions, activation_fn=None)
                out_list.append(out_temp)
            
        return out_list
