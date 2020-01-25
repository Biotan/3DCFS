# -*- coding:utf-8 -*-
from cell import CFSM
import tensorflow as tf
from train import FLAGS, GRAPH

timesteps = 2
shape = [4096, 1]
kernel = [1, 1]
channels = 128
filters = 128

with GRAPH.device('/gpu:' + str(FLAGS.gpu)):
    tf.variable_scope('cell_ins')
    tf.variable_scope('cell_seg')
with tf.variable_scope('cell_ins'):
    cell_ins = CFSM(shape, filters, kernel)
    # cell_ins = ConvGRUCell(shape, filters, kernel)
with tf.variable_scope('cell_seg'):
    cell_seg = CFSM(shape, filters, kernel)
    # cell_seg = ConvGRUCell(shape, filters, kernel)


def twin_cfsm(X_Ins, X_Seg):
    X_Ins = tf.expand_dims(X_Ins, 1)
    X_Seg = tf.expand_dims(X_Seg, 1)

    X_seg_ins = tf.concat([X_Seg, X_Ins], 1)
    X_ins_seg = tf.concat([X_Ins, X_Seg], 1)

    X_seg_ins = tf.expand_dims(X_seg_ins, 4)
    X_ins_seg = tf.expand_dims(X_ins_seg, 4)

    # X_ins_seg ===> [24,2,4096,1,128]  X_seg_ins ===> [24,2,4096,1,128]
    X_seg_ins = tf.transpose(X_seg_ins, [0, 1, 2, 4, 3])
    X_ins_seg = tf.transpose(X_ins_seg, [0, 1, 2, 4, 3])

    with tf.variable_scope('cell_ins'):
        outputs_seg_ins, state_seg_ins = tf.nn.dynamic_rnn(cell_ins, X_seg_ins, dtype=X_seg_ins.dtype)

    with tf.variable_scope('cell_seg'):
        outputs_ins_seg, state_ins_seg = tf.nn.dynamic_rnn(cell_seg, X_ins_seg, dtype=X_ins_seg.dtype)

    # [batch,steps,w,h,c] ===> [steps,batch,w,h,c]
    outputs_seg_ins = tf.transpose(outputs_seg_ins, [1, 0, 2, 3, 4])[1]
    outputs_ins_seg = tf.transpose(outputs_ins_seg, [1, 0, 2, 3, 4])[1]
    Result_seg_ins = tf.squeeze(outputs_seg_ins, 2)
    Result_ins_seg = tf.squeeze(outputs_ins_seg, 2)
    return Result_seg_ins, Result_ins_seg

