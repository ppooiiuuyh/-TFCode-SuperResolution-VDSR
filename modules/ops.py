import tensorflow as tf
import tensorflow.contrib as tf_contrib
import math


EPS = 1e-8


def instance_normalization(x, name):
    with tf.variable_scope(name):
        depth = x.get_shape().as_list()[-1]
        scale = tf.get_variable("insnorm_scale", [depth],
                                initializer=tf.random_normal_initializer(
                                    mean=1.0, stddev=0.02, dtype=tf.float32))
        offset = tf.get_variable("insnorm_offset", [depth],
                                 initializer=tf.random_normal_initializer(
                                     mean=0.0, stddev=0.02, dtype=tf.float32))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-8
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg


def adaptive_instance_norm(x, mu, sigma):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    inv = tf.rsqrt(variance + EPS)
    return sigma * (x - mean) * inv + mu


def residual_with_adaptive_insnorm(x, mu, sigma, name=None, ksize=3,
                                   activation=leaky_relu, norm_type=None, is_training=False, wd=None,
                                   reuse=None, initializer=tf.contrib.layers.variance_scaling_initializer,
                                   update_collection=None):
    fdim = x.get_shape().as_list()[-1]
    if wd is None:  # Weight decay
        kr = None
    else:
        kr = tf.contrib.layers.l2_regularizer(scale=wd)

    with tf.variable_scope(name, reuse=reuse) as scope:
        y = conv_block(x, "conv0", ksize=ksize, strides=1, fsize=fdim,
                       activation=None, norm_type=None, is_training=is_training,
                       wd=wd, initializer=initializer, update_collection=update_collection,
                       reuse=reuse)
        y = adaptive_instance_norm(y, mu, sigma)

        if activation is not None:
            y = activation(y)
        y = conv_block(y, "conv1", ksize=ksize, strides=1, fsize=fdim,
                       activation=None, norm_type=None, is_training=is_training,
                       wd=wd, initializer=initializer, update_collection=update_collection,
                       reuse=reuse)
        y = adaptive_instance_norm(y, mu, sigma)

        return x + y


def residual_block(x, name=None, ksize=3, activation=leaky_relu,
                   norm_type=['spectral_norm'], is_training=False, wd=None, reuse=None,
                   initializer=tf.contrib.layers.variance_scaling_initializer,
                   update_collection=None):
    fdim = x.get_shape().as_list()[-1]

    with tf.variable_scope(name, reuse=reuse) as scope:
        y = conv_block(x, "conv0", ksize=ksize, strides=1, fsize=fdim,
                       activation=activation, norm_type=norm_type, is_training=is_training,
                       wd=wd, initializer=initializer, update_collection=update_collection,
                       reuse=reuse)
        y = conv_block(y, "conv1", ksize=ksize, strides=1, fsize=fdim,
                       activation=None, norm_type=norm_type, is_training=is_training,
                       wd=wd, initializer=initializer, update_collection=update_collection,
                       reuse=reuse)

        return x + y


def non_local_block(x, name=None, ksize=3, activation=leaky_relu,
                    norm_type=['spectral_norm'], is_training=False, wd=0.0001, reuse=None,
                    initializer=tf.contrib.layers.variance_scaling_initializer,
                    update_collection=None):
    BS, H, W, fdim = x.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse) as scope:
        # theta
        theta = conv_block(x, "conv0", ksize=1, strides=1, fsize=fdim / 8,
                           activation=activation, norm_type=norm_type, is_training=is_training,
                           wd=wd, initializer=initializer, update_collection=update_collection,
                           reuse=reuse)
        theta = tf.reshape(theta, [BS, -1, theta.get_shape()[-1]])

        # phi
        phi = conv_block(x, "conv1", ksize=1, strides=1, fsize=fdim / 8,
                         activation=None, norm_type=norm_type, is_training=is_training,
                         wd=wd, initializer=initializer, update_collection=update_collection,
                         reuse=reuse)
        phi = tf.reshape(phi, [BS, -1, phi.get_shape()[-1]])
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # g path
        g = conv_block(x, "conv2", ksize=1, strides=1, fsize=fdim / 2,
                       activation=None, norm_type=norm_type, is_training=is_training,
                       wd=wd, initializer=initializer, update_collection=update_collection,
                       reuse=reuse)
        g = tf.reshape(g, [BS, -1, g.get_shape()[-1]])

        attn_g = tf.matmul(attn, g)
        if is_training:
            attn_g = tf.reshape(attn_g, [BS, H, W, -1])
        else:
            attn_g = tf.reshape(attn_g, [
                BS, tf.shape(x)[1], tf.shape(x)[2], attn_g.get_shape()[-1]])
        sigma = tf.get_variable(
            'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_g = conv_block(attn_g, "conv3", ksize=1, strides=1, fsize=fdim,
                            activation=None, norm_type=norm_type, is_training=is_training,
                            wd=wd, initializer=initializer,
                            update_collection=update_collection,
                            reuse=reuse)

        return x + sigma * attn_g


def resize(x, scale=2.0):
    return tf.image.resize_images(
        x, [tf.cast(scale * tf.cast(tf.shape(x)[1], tf.float32), tf.int32),
            tf.cast(scale * tf.cast(tf.shape(x)[2], tf.float32), tf.int32)],
        align_corners=True,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



def conv2D(x, name=None, ksize=3, strides=2, fsize=64, activation=None,
               norm_type=None, is_training=False, wd=None, padding='SAME',
               initializer=tf.contrib.layers.xavier_initializer,
               update_collection=None, dilations=[1, 1, 1, 1], use_self_attention = False, shape=None):
    with tf.variable_scope(name):
        """ 1. set weight """
        if type(ksize) is int: ksize = [ksize,ksize]
        w = tf.get_variable("kernel", [ksize[0], ksize[1], x.get_shape()[-1], fsize],   initializer=initializer)

        if norm_type is not None and 'spectral_norm' in norm_type:
            w = spectral_norm(w)


        """ 2. build convolutional layer """
        y = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
        biases = tf.get_variable('bias', [fsize], initializer=tf.constant_initializer(0.0))
        y = y + biases

        """ 3. apply normalization """
        #  if norm_type is not None:
        if norm_type is not None:
            if 'instance_norm' in norm_type:
                y = instance_normalization(y, "insnorm")
            elif 'batch_norm' in norm_type:
                y = tf.layers.batch_normalization(y, center=True, scale=True,
                                                  training=is_training)
            elif 'group_norm' in norm_type:
                y = tf.contrib.layers.group_norm(y)


        """ 4. apply activation function """
        if activation is not None:
            y = activation(y)


        """ 5. apply self attention """
        if use_self_attention:
            y = self_attention(y,channels=fsize, name=name, shape = shape)

        return y



def deconv(x,  ksize, filters, strides, name,  padding="SAME", dilations=[1,1,1,1], shape = None, activation = None):
    with tf.variable_scope(name):
        if shape is None:
            weight_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable("kernel", [ksize, ksize, filters, x.shape[-1]],initializer=weight_init)

            x = tf.nn.conv2d_transpose(x,filters=w, strides=[1,strides,strides,1], output_shape=[x.shape[0],x.shape[1]*strides,x.shape[2]*strides,filters],
                                       padding = padding, name = name, dilations = dilations)

        else:
            x = tf.nn.conv2d_transpose(x, filter=[ksize, ksize, shape[-1], filters], output_shape=[shape[0], shape[1]*strides, shape[2] * strides, filters],
                                       padding=padding, name=name, dilations=dilations)

        if activation is not None:
            x = activation(x)
    return x




def fc(x, units, activation=None, name=None, reuse = tf.AUTO_REUSE):
    wi = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name, reuse = reuse):
        x = tf.layers.dense(x, units, activation=activation,
                        kernel_initializer=wi, name=name)
    return x






def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, axis=1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, b, axis=1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def subpixel_upsample(X, r, num_channels, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse) as scope:
        bsize, a, b, c = X.get_shape().as_list()
        assert (c == num_channels * r * r)
        Xc = tf.split(X, num_channels, axis=3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
        bsize, a, b, c = X.get_shape().as_list()
        assert (c == num_channels)


    return X


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]

def flatten(x):
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1])  # return [BATCH, W*H, CHANNELS]


def self_attention(x, channels, use_bias=True, sn=False, name='self_attention',shape = None):
    with tf.variable_scope(name):
        factor = 8
        f = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='f_conv')  # [bs, h, w, c']
        g = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='g_conv')  # [bs, h, w, c']
        h = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='h_conv')  # [bs, h, w, c']

        # N = h * w
        num_grid = 64
        pooling_size = max(x.shape.as_list()[1]//num_grid,1)
        g = tf.nn.max_pool2d(g, ksize=[1,pooling_size,pooling_size,1], padding="VALID", strides=[1,pooling_size,pooling_size,1])
        f = tf.nn.max_pool2d(f, ksize=[1,pooling_size,pooling_size,1], padding="VALID", strides=[1,pooling_size,pooling_size,1])
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        s = tf.reduce_sum(s,axis=-1)
        beta = s
        #beta=tf.nn.sigmoid(s)
        #beta = tf.nn.softmax(s)  # attention map
        #beta = tf.nn.sigmoid(s)
        #beta = tf.exp(s) / tf.reduce_sum(tf.exp(s),[1,2])
        if shape is None:
            beta = tf.reshape(beta, shape=[x.shape[0],x.shape[1]//pooling_size,x.shape[2]//pooling_size,1])  # [bs, h, w, C]
        else:
            beta = tf.reshape(beta, shape=[shape[0],shape[1]//pooling_size,shape[2]//pooling_size,1])  # [bs, h, w, C]
        beta = resize(beta, scale = pooling_size)
        beta = tf.nn.sigmoid(beta)

        o = beta * h
        o = conv_block(o, fsize = channels, ksize=1, strides=1, name ='v_conv')  # [bs, h, w, c]
        x = o + x



        #gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        #x = gamma * o + x

    return x, o, o

"""
def self_attention(x, channels, k = 7, use_bias=True, sn=False, name='self_attention',shape = None):
    with tf.variable_scope(name):
        y = x
        y1 = conv_block(y, fsize = channels//8, ksize=[1,k], strides=1, name ='f1_conv')  # [bs, h, w, c']
        y1 = conv_block(y1, fsize = 1, ksize=[k,1], strides=1, name ='g1_conv')  # [bs, h, w, c']
        #y2 = conv_block(y, fsize = channels//8, ksize=[k,1], strides=1, name ='g2_conv')  # [bs, h, w, c']
        #y2 = conv_block(y2, fsize = 1, ksize=[1,k], strides=1, name ='f2_conv')  # [bs, h, w, c']
        #y = y1+y2
        y = tf.nn.sigmoid(y)
        x = y*x
    return x, y, y

def self_attention(x, channels, use_bias=True, sn=False, name='self_attention',shape = None):
    with tf.variable_scope(name):
        factor = 16
        f = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='f_conv')  # [bs, h, w, c']
        g = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='g_conv')  # [bs, h, w, c']
        h = conv_block(x, fsize = channels // factor, ksize=1, strides=1, name ='h_conv')  # [bs, h, w, c']

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map
        #beta = tf.nn.sigmoid(s)
        #beta = tf.exp(s) / tf.reduce_sum(tf.exp(s),[1,2])

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        if shape is None:
            o = tf.reshape(o, shape=[x.shape[0],x.shape[1],x.shape[2],channels//factor])  # [bs, h, w, C]
        else:
            o = tf.reshape(o, shape=shape)  # [bs, h, w, C]
        o = conv_block(o, fsize = channels, ksize=1, strides=1, name ='v_conv')  # [bs, h, w, c]


        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        x = gamma * o + x

    return x, gamma, o
"""


def self_attention_with_pooling(x, channels, use_bias=True, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv_block(x, fsize = channels // 8, ksize=1, strides=1, name ='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv_block(x, fsize = channels // 8, ksize=1, strides=1, name ='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='attn_conv')
        x = gamma * o + x

    return x


def self_attention_CBAM(x, channels, name='CBAM', shape = None):
    with tf.variable_scope(name):
        """ cal MC """
        ap = tf.nn.avg_pool2d(x,ksize = [1,x.shape[1], x.shape[2],1], strides=[1,1,1,1], padding = "VALID")
        mp = tf.nn.max_pool2d(x,ksize = [1,x.shape[1], x.shape[2],1], strides=[1,1,1,1], padding = "VALID")

        ap = tf.reshape(ap,shape=[x.shape[0], x.shape[3]])
        mp = tf.reshape(mp,shape=[x.shape[0], x.shape[3]])

        ap_fc = fc(fc(ap,channels//16, name = "MC_fc1"), channels, name="MC_fc2")
        mp_fc = fc(fc(mp,channels//16, name = "MC_fc1"), channels, name="MC_fc2")

        MC = tf.nn.sigmoid(ap_fc + mp_fc)
        MC = tf.reshape(MC, shape=[x.shape[0], 1, 1, x.shape[3]])

        """ cal MS """
        MS = conv_block(x,name = "MS_conv1", ksize =1, strides=1, fsize = 1)
        MS = conv_block(MS,name = "MS_conv2", ksize = 3, strides=1, fsize = 1, dilations=[1,4,4,1])
        MS = tf.nn.sigmoid(MS)

        y = MC * x
        y = MS * y+ x
    return y


def squeeze_excitation(x, channels, ratio=16, use_bias=True, sn=False, scope='senet'):
    with tf.variable_scope(scope):
        squeeze = global_avg_pooling(x)

        excitation = fully_connected(squeeze, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
        excitation = relu(excitation)
        excitation = fully_connected(excitation, units=channels, use_bias=use_bias, sn=sn, scope='fc2')
        excitation = sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, channels])

        scale = x * excitation

        return scale

def convolution_block_attention(x, channels, ratio=16, use_bias=True, sn=False, scope='cbam') :
    with tf.variable_scope(scope) :
        with tf.variable_scope('channel_attention') :
            x_gap = global_avg_pooling(x)
            x_gap = fully_connected(x_gap, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
            x_gap = relu(x_gap)
            x_gap = fully_connected(x_gap, units=channels, use_bias=use_bias, sn=sn, scope='fc2')

        with tf.variable_scope('channel_attention', reuse=True):
            x_gmp = global_max_pooling(x)
            x_gmp = fully_connected(x_gmp, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
            x_gmp = relu(x_gmp)
            x_gmp = fully_connected(x_gmp, units=channels, use_bias=use_bias, sn=sn, scope='fc2')

            scale = tf.reshape(x_gap + x_gmp, [-1, 1, 1, channels])
            scale = sigmoid(scale)

            x = x * scale

        with tf.variable_scope('spatial_attention') :
            x_channel_avg_pooling = tf.reduce_mean(x, axis=-1, keepdims=True)
            x_channel_max_pooling = tf.reduce_max(x, axis=-1, keepdims=True)
            scale = tf.concat([x_channel_avg_pooling, x_channel_max_pooling], axis=-1)

            scale = conv(scale, channels=1, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, sn=sn, scope='conv')
            scale = sigmoid(scale)

            x = x * scale

            return x

def global_context_block(x, channels, use_bias=True, sn=False, scope='gc_block'):
    with tf.variable_scope(scope):
        with tf.variable_scope('context_modeling'):
            bs, h, w, c = x.get_shape().as_list()
            input_x = x
            input_x = hw_flatten(input_x)  # [N, H*W, C]
            input_x = tf.transpose(input_x, perm=[0, 2, 1])
            input_x = tf.expand_dims(input_x, axis=1)

            context_mask = conv(x, channels=1, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv')
            context_mask = hw_flatten(context_mask)
            context_mask = tf.nn.softmax(context_mask, axis=1)  # [N, H*W, 1]
            context_mask = tf.transpose(context_mask, perm=[0, 2, 1])
            context_mask = tf.expand_dims(context_mask, axis=-1)

            context = tf.matmul(input_x, context_mask)
            context = tf.reshape(context, shape=[bs, 1, 1, c])

        with tf.variable_scope('transform_0'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_0')
            context_transform = layer_norm(context_transform)
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_1')
            context_transform = sigmoid(context_transform)

            x = x * context_transform

        with tf.variable_scope('transform_1'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_0')
            context_transform = layer_norm(context_transform)
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_1')

            x = x + context_transform

        return x