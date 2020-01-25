import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from loss import *
import cfsm


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    # l0_xyz [12 4096 3]
    l0_points = point_cloud[:, :, 3:]
    # l0_points [12 4096 6]
    end_points['l0_xyz'] = l0_xyz

    input_image = tf.expand_dims(point_cloud, -1)
    # CONV
    net = tf_util.conv2d(input_image, 64, [1, 9], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                                  bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # print(pc_feat1)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net = tf_util.conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv7')
    #net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
    #                     bn=True, is_training=is_training, scope='conv8')

    net = tf.squeeze(net, 2)
    net_sem = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc1',
                             bn_decay=bn_decay)

    net_ins = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc1',
                             bn_decay=bn_decay)
    #net_sem = tf_util.conv1d(net_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc2',
    #                         bn_decay=bn_decay)

    #net_ins = tf_util.conv1d(net_ins, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc2',
    #                         bn_decay=bn_decay)

    net_ins, net_sem = cfsm.twin_cfsm(net_ins, net_sem)

    net_ins = tf_util.conv1d(net_ins, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc0',
                             bn_decay=bn_decay)
    net_sem = tf_util.conv1d(net_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc0',
                             bn_decay=bn_decay)

    net_ins = tf_util.dropout(net_ins, keep_prob=0.5, is_training=is_training, scope='ins_dp1')
    net_ins = tf_util.conv1d(net_ins, 5, 1, padding='VALID', activation_fn=None, scope='ins_fc4')
    net_sem = tf_util.dropout(net_sem, keep_prob=0.5, is_training=is_training, scope='sem_dp1')
    net_sem = tf_util.conv1d(net_sem, num_class, 1, padding='VALID', activation_fn=None, scope='sem_fc4')

    return net_sem, net_ins  # net_sem [12 4096 13]  net_ins [12 4096 5]


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape().as_list()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.001

    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, feature_dim,
                                                          delta_v, delta_d, param_var, param_dist, param_reg)

    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist, l_reg


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 2048, 3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
