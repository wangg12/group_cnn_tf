from __future__ import division, absolute_import, print_function
import tensorflow as tf
# keras = tf.contrib.keras
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

def conv_bn_act(x, data_format='NHWC', Cout=20, ksize=3, stride=1,
                bn=True, is_train=True,
                act=tf.nn.relu,
                reuse=False, name='conv_bn_act'):
  with tf.variable_scope(name, reuse=reuse) as vs:
    conv = tf.layers.conv2d(
        inputs=x,
        filters=Cout,
        kernel_size=[ksize, ksize],
        padding="valid",
        activation=None, reuse=reuse, name='conv2d')
    if bn:
      conv = tf.layers.batch_normalization(conv, training=is_train)
    return act(conv)

# def conv_bn_act(x, data_format='channels_last', Cout=20, ksize=3, stride=1,
#                 bn=True, is_train=True,
#                 act=tf.nn.relu):
#   x = keras.layers.Conv2D(
#       filters=Cout,
#       kernel_size=[ksize, ksize], strides=(stride, stride),
#       padding="valid",
#       data_format='channels_last',
#       activation=None)(x)
#   if bn:
#     x = keras.layers.BatchNormalization(axis=-1)(x, training=is_train)
#   return act(x)

def z2_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False):
  '''
  x: NHWC
  y_: N
  '''
  C_out = 20
  with tf.variable_scope("Z2CNN", reuse=reuse) as vs:
    conv_bn_act_1 = conv_bn_act(x, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act1')
    drop1 = tf.layers.dropout(inputs=conv_bn_act_1, rate=drop_rate,
                training=is_train, name='drop1')

    conv_bn_act_2 = conv_bn_act(drop1, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act2')
    max_pool_2 = tf.layers.max_pooling2d(conv_bn_act_2, pool_size=2, strides=2,
                    padding='valid',
                    data_format='channels_last', name='max_pool2')

    conv_bn_act_3 = conv_bn_act(max_pool_2, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act3')
    drop3 = tf.layers.dropout(inputs=conv_bn_act_3, rate=drop_rate,
                training=is_train, name='drop3')

    conv_bn_act_4 = conv_bn_act(drop3, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act4')
    drop4 = tf.layers.dropout(inputs=conv_bn_act_4, rate=drop_rate,
                training=is_train, name='drop4')

    conv_bn_act_5 = conv_bn_act(drop4, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act5')
    drop5 = tf.layers.dropout(inputs=conv_bn_act_5, rate=drop_rate,
                training=is_train, name='drop5')

    conv_bn_act_6 = conv_bn_act(drop5, Cout=C_out, is_train=is_train,
                reuse=reuse, name='conv_bn_act6')
    drop6 = tf.layers.dropout(inputs=conv_bn_act_6, rate=drop_rate,
                training=is_train, name='drop6')

    conv_7 = tf.layers.conv2d(inputs=drop6,
                filters=10,
                kernel_size=[4, 4],
                padding="valid",
                activation=None, reuse=reuse, name='conv7')
    # print(conv_7.get_shape()) # (?, 1, 1, 10)

    # take max value of (H,W) dimensions
    # NHWC --> Nx10
    logits = tf.reduce_max(conv_7, axis=(1, 2), keep_dims=False, name='reduce_max1')
    
  return logits



def gconv_bn_act(x, gconv_type, C_in, C_out, ksize=3, padding='VALID', 
                 bn=True, is_train=True, act=tf.nn.relu, 
                 name='gconv_bn_act', reuse=False):
    if gconv_type == 'Z2_to_P4':
        h_in = 'Z2'
        h_out = 'C4'
    elif gconv_type == 'Z2_to_P4M':
        h_in = 'Z2'
        h_out = 'D4'
    elif gconv_type == 'P4_to_P4':
        h_in = 'C4'
        h_out = 'C4'
    elif gconv_type == 'P4M_to_P4M':
        h_in = 'D4'
        h_out = 'D4'
    else:
        raise NotImplemented('Unsupported gconv_type: {}!'.format(gconv_type))
    with tf.variable_scope(name, reuse=reuse) as vs:
        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
                h_input=h_in, h_output=h_out, in_channels=C_in, out_channels=C_out, ksize=ksize)
        # w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.), name='kernel')
        w = tf.get_variable("kernel", shape=w_shape, 
                    initializer=tf.contrib.layers.xavier_initializer()) # initialization is very important!!!
        x = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding=padding,
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
        if bn:
            x = tf.layers.batch_normalization(x, training=is_train)
        return act(x)



def p4_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False):
  '''
  x: NHWC
  y_: N
  '''
  C_out = 10
  with tf.variable_scope("P4CNN", reuse=reuse) as vs:
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=1, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act1') # l1
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                 training=is_train, name='drop1')
    # print(x.get_shape().as_list()) # [None, 26, 26, 40]

    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4_to_P4_gconv_bn_act2') # l2
    # print(x.get_shape().as_list()) # [None, 24, 24, 40]
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,
                    padding='valid',
                    data_format='channels_last', name='max_pool2')
    # print(x.get_shape().as_list()) # [None, 12, 12, 40]
    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4_to_P4_gconv_bn_act3') # l3
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop3')
    # print(x.get_shape().as_list()) # [None, 10, 10, 40]
    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4_to_P4_gconv_bn_act4') # l4
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop4')
    # print(x.get_shape().as_list()) # [None, 8, 8, 40]
    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4_to_P4_gconv_bn_act5') # l5
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop5')
    # print(x.get_shape().as_list()) # [None, 6, 6, 40]
    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4_to_P4_gconv_bn_act6') # l6
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop6')
    # print(x.get_shape().as_list()) # [None, 4, 4, 40]
    x = gconv_bn_act(x, gconv_type='P4_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                padding="VALID",
                bn=False,
                act=tf.identity, reuse=reuse, name='P4_to_P4_gconv7',) # l7
    # print(x.get_shape().as_list()) # [None, 2, 2, 40]

    # take max value of (H,W) dimensions
    # NHWC --> Nx10
    xs = x.get_shape().as_list() 
    x = tf.reshape(x, shape=[-1, xs[1], xs[2], 4, xs[3]//4])
    # print(x.get_shape().as_list()) # [None, 2, 2, 4, 10]
    logits = tf.reduce_max(x, axis=(1, 2, 3), keep_dims=False, name='reduce_max1')
    # print(logits.get_shape().as_list()) # [None, 10]
  return logits



def p4m_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False):
  '''
  x: NHWC
  y_: N
  '''
  C_out = 10
  with tf.variable_scope("P4MCNN", reuse=reuse) as vs:
    x = gconv_bn_act(x, gconv_type='Z2_to_P4M', C_in=1, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4M_gconv_bn_act1') # l1
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                 training=is_train, name='drop1')
    # print(x.get_shape().as_list()) # [None, 26, 26, 80]

    x = gconv_bn_act(x, gconv_type='P4M_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4M_to_P4M_gconv_bn_act2') # l2
    # print(x.get_shape().as_list()) # [None, 24, 24, 80]
    xs = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, xs[1], xs[2], 8, xs[3]//8])
    # print(x.get_shape().as_list()) # [None, 24, 24, 8, 10]
    x = tf.reduce_max(x, axis=(-2), keep_dims=False, name='reduce_max_l2')
    # print(x.get_shape().as_list()) # [None, 24, 24, 10]
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,
                    padding='valid',
                    data_format='channels_last', name='max_pool2')
    # print(x.get_shape().as_list()) # [None, 12, 12, 10]
    x = gconv_bn_act(x, gconv_type='Z2_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4M_gconv_bn_act3') # l3
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop3')
    # print(x.get_shape().as_list()) # [None, 10, 10, 80]
    x = gconv_bn_act(x, gconv_type='P4M_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4M_to_P4M_gconv_bn_act4') # l4
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop4')
    # print(x.get_shape().as_list()) # [None, 8, 8, 80]
    x = gconv_bn_act(x, gconv_type='P4M_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4M_to_P4M_gconv_bn_act5') # l5
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop5')
    # print(x.get_shape().as_list()) # [None, 6, 6, 80]
    x = gconv_bn_act(x, gconv_type='P4M_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='P4M_to_P4M_gconv_bn_act6') # l6
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop6')
    # print(x.get_shape().as_list()) # [None, 4, 4, 80]
    x = gconv_bn_act(x, gconv_type='P4M_to_P4M', C_in=C_out, C_out=C_out, is_train=is_train,
                padding="VALID",
                bn=False,
                act=tf.identity, reuse=reuse, name='P4M_to_P4M_gconv7',) # l7
    # print(x.get_shape().as_list()) # [None, 2, 2, 80]

    # take max value of (H,W) dimensions
    # NHWC --> Nx10
    xs = x.get_shape().as_list() 
    x = tf.reshape(x, shape=[-1, xs[1], xs[2], 8, xs[3]//8])
    # print(x.get_shape().as_list()) # [None, 2, 2, 8, 10]
    # logits = tf.reduce_max(x, axis=(1, 2, 3), keep_dims=False, name='reduce_max_l7')
    logits = tf.reduce_mean(x, axis=(1, 2, 3), keep_dims=False, name='reduce_mean_l7')
    # print(logits.get_shape().as_list()) # [None, 10]
  return logits


def rotation_pooling(x, pool_type='max', name='rotation_pooling', reuse=False):
  '''
  type: 'max' or 'mean'/'average'
  '''
  with tf.variable_scope(name, reuse):
    xs = x.get_shape().as_list()  # [B, H, W, S, C]
    x = tf.reshape(x, shape=[-1, xs[1], xs[2], 4, xs[3]//4])
    if pool_type == 'max':
      x = tf.reduce_max(x, axis=-2, keep_dims=False)
    elif pool_type == 'mean' or pool_type == 'average':
      x = tf.reduce_mean(x, axis=-2, keep_dims=False)
    return x



def p4_cnn_rp(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False):
  '''P4CNN with Rotation Pooling
  x: NHWC
  y_: N
  '''
  C_out = 20
  rp_type = 'average' # 'max'
  with tf.variable_scope("P4CNN_RP", reuse=reuse) as vs:
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=1, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act1') # l1
    
    # print(x.get_shape().as_list()) # [None, 26, 26, 80]

    x = rotation_pooling(x, pool_type=rp_type, name='RP_1')
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                 training=is_train, name='drop1')
    # print(x.get_shape().as_list()) # [None, 26, 26, 20]

    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act2') # l2
    # print(x.get_shape().as_list()) # [None, 24, 24, 80]
    x = rotation_pooling(x, pool_type=rp_type, name='RP_2')
    # print(x.get_shape().as_list()) # [None, 24, 24, 20]
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2,
                    padding='valid',
                    data_format='channels_last', name='max_pool2')
    # print(x.get_shape().as_list()) # [None, 12, 12, 20]
    
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act3') # l3
    # print(x.get_shape().as_list()) # [None, 10, 10, 80]
    x = rotation_pooling(x, pool_type=rp_type, name='RP_3')
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop3')
    # print(x.get_shape().as_list()) # [None, 10, 10, 20]
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act4') # l4
    # print(x.get_shape().as_list()) # [None, 8, 8, 80]
    x = rotation_pooling(x, pool_type=rp_type, name='RP_4')
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop4')
    # print(x.get_shape().as_list()) # [None, 8, 8, 20]
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act5') # l5
    # print(x.get_shape().as_list()) # [None, 6, 6, 80]
    x = rotation_pooling(x, pool_type=rp_type, name='RP_5')
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop5')
    # print(x.get_shape().as_list()) # [None, 6, 6, 20]
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=C_out, is_train=is_train,
                reuse=reuse, name='Z2_to_P4_gconv_bn_act6') # l6
    # print(x.get_shape().as_list()) # [None, 4, 4, 80]
    x = rotation_pooling(x, pool_type=rp_type, name='RP_6')
    x = tf.layers.dropout(inputs=x, rate=drop_rate,
                training=is_train, name='drop6')
    # print(x.get_shape().as_list()) # [None, 4, 4, 20]
    x = gconv_bn_act(x, gconv_type='Z2_to_P4', C_in=C_out, C_out=10, ksize=4, is_train=is_train,
                padding="VALID",
                bn=False,
                act=tf.identity, reuse=reuse, name='P4_to_P4_gconv7',) # l7
    print(x.get_shape().as_list()) # [None, 1, 1, 40]

    # take max value of (H,W) dimensions
    # NHWC --> Nx10
    xs = x.get_shape().as_list() 
    x = tf.reshape(x, shape=[-1, xs[1], xs[2], 4, xs[3]//4])
    # print(x.get_shape().as_list()) # [None, 1, 1, 4, 10]
    logits = tf.reduce_max(x, axis=(1, 2, 3), keep_dims=False, name='reduce_max_l7')
    # logits = tf.reduce_mean(x, axis=(1, 2, 3), keep_dims=False, name='reduce_mean_l7')
    # print(logits.get_shape().as_list()) # [None, 10]
  return logits

if __name__ == '__main__':
  
  # import numpy as np
  x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'x')
  y_ = tf.placeholder(tf.int32, [None], 'y_')
  # is_train = tf.placeholder(tf.bool, name='is_train')
  # # keep_prob = tf.placeholder(tf.float32, name='keep_prob') # no need when use tf.layers.dropout

  # # create fake test data
  # xx = np.ones((3, 28, 28, 1)).astype(np.float32)
  # tt = np.ones(3).astype(np.int32)

  logits = p4_cnn_rp(x, y_, is_train=True)

  # # count number of trainable parameters
  # num_params = np.sum([np.prod(v.get_shape().as_list())
  #                       for v in tf.trainable_variables()])
  # print('num_params: {}'.format(num_params))

  # with tf.Session() as sess:
  #   init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  #   sess.run(init_op)
  #   logits0, loss0, acc0 = sess.run([logits, loss, acc], feed_dict={x:xx, y_:tt, is_train:True})
  #   print('logits shape:', logits0.shape)
  #   print('loss:', loss0)
  #   print('acc:', acc0)

  #   print('test phase')
  #   logits0, loss0, acc0 = sess.run([logits, loss, acc], feed_dict={x:xx, y_:tt, is_train:False})
  #   print('logits shape:', logits0.shape)
  #   print('loss:', loss0)
  #   print('acc:', acc0)














