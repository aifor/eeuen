import tensorflow as tf
import numpy as np
import sys
import math

## initializer
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

## re-package unit
def bn(data, is_training, center=True, scale=True, epsilon=1e-3, momentum=0.9, scope=None):
    print("# %s #"%tf.get_default_graph().get_name_scope())
    out = tf.layers.batch_normalization(
        inputs=data,
        momentum=momentum,
        epsilon=epsilon,
        scale=scale,
        moving_variance_initializer=tf.zeros_initializer(),
        training=is_training,
        center=center,
        reuse=tf.AUTO_REUSE,
        name=scope
    )
    return out

## activate function
def prelu(x, trainable = True):
    # x = tf.convert_to_tensor(x)
    alpha = tf.get_variable(
        'prelu_alpha',
        shape=(x.get_shape()[-1],),
        initializer=tf.constant_initializer(0.1),
        trainable = trainable
    )
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def dice(x, is_training = True, trainable = True):
    # x = tf.convert_to_tensor(x)
    out = bn(x, is_training=is_training, center=False, scale=False, epsilon=1e-4, momentum=0.99, scope='dice_bn')
    logits = tf.nn.sigmoid(out)  # 1 / (1 + tf.exp(-out))
    dice_gamma = tf.get_variable(
        'dice_gamma',
        shape=(1, x.get_shape()[-1]),
        initializer=tf.constant_initializer(0.1),
        trainable=trainable
    )
    return tf.multiply(dice_gamma, (1.0 - logits) * x) + logits * x

def swish(x):
    return x * tf.nn.sigmoid(x)

def h_swish(x):
    return x * tf.nn.relu6(x + tf.constant(3.)) / 6.

def pswish(x, trainable = True):
    beta = tf.get_variable(
        'pswish_beta',
        shape=(x.get_shape()[-1],),
        initializer=tf.constant_initializer(1.0),
        trainable=trainable
    )
    return x * tf.nn.sigmoid(beta * x)

def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))

def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf

def gelu2(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2. / 3.1415926) * (x + 0.044715 * tf.pow(x, 3))))

def fc(data, units, act_type, is_training, trainable = True, l2_reg = None,
       use_bn = False, drop_rate = None, init_w_type = None, scope=None):
    units_in = int(data.shape[-1])
    assert units_in > 0, 'invalid input shape: %s'%data.shape

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if init_w_type is None:
            # Kaiming initializing
            weight_value = np.random.randn(units_in, units).astype(
                np.float32
            ) * np.sqrt(2. / units_in)
            # weight_value = np.transpose(weight_value)
            w_init = tf.initializers.constant(weight_value)
        elif init_w_type == 'norm':
            w_init = tf.initializers.random_normal(mean=0.0, stddev=0.01 / units_in)
        elif init_w_type == 'trunc_norm':
            w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.02 / units_in)
        elif init_w_type == 'xavier':
            w_init = xavier_init(units_in, units, False)
        elif init_w_type == 'xavier_n':
            w_init = xavier_init(units_in, units, True)
        else:
            w_init = tf.initializers.random_uniform(minval=0.)

        #print scope, "############", weight_value.shape, weight_value
        print("# %s/fc #"%tf.get_default_graph().get_name_scope())

        kener_reg = None
        if l2_reg is not None:
            kener_reg = tf.contrib.layers.l2_regularizer(l2_reg)
        dout = tf.layers.dense(data, units, activation = None,
                               kernel_initializer= w_init,
                               bias_initializer=tf.initializers.constant(0.1),
                               kernel_regularizer = kener_reg,
                               name='fc',
                               trainable=trainable)

        if use_bn and act_type != 'dice':
            dout = bn(dout, is_training, scope='bn')

        if act_type == 'sigmoid':
            out = tf.nn.sigmoid(dout)
        elif act_type == 'tanh':
            out = tf.nn.tanh(dout)
        elif act_type == 'relu':
            out = tf.nn.relu(dout)
        elif act_type == "elu":
            out = tf.nn.elu(dout)
        elif act_type == "softplus":
            out = tf.nn.softplus(dout)
        elif act_type == 'prelu':
            out = prelu(dout, trainable)
        elif act_type == 'dice':
            out = dice(dout, is_training, trainable)
        elif act_type == 'swish':
            out = swish(dout)
        elif act_type == 'h_swish':
            out = h_swish(dout)
        elif act_type == 'pswish':
            out = pswish(dout, trainable)
        elif act_type == 'mish':
            out = mish(dout)
        elif act_type == 'tanh':
            out = tf.nn.tanh(dout)
        elif act_type == 'gelu':
            out = gelu(dout)
        elif act_type == 'gelu2':
            out = gelu2(dout)
        elif not act_type:
            out = dout
        else:
            raise RuntimeError('unknown act_type %s' % act_type)

        if drop_rate is not None:
            out = tf.layers.dropout(out, drop_rate, training = is_training, name = 'drop')

    return out

def fc_repeats(data, shapes, is_training, trainable = True, acts=None, weights=None):
    assert weights is None or len(weights) == len(shapes)
    assert acts is None or len(acts) == len(shapes)

    units_in = int(data.shape[-1])
    assert units_in > 0, 'invalid input shape: %s'%data.shape
    dout = data
    for l in range(len(shapes)):
        #print ">>>>>>>>>>>>>>>>>", dout.shape.as_list()
        if weights is None or weights[l] is None or weights[l] is '':
            weight = np.random.randn(units_in, shapes[l]).astype(
                np.float32
            ) * np.sqrt(2. / units_in)
        else:
            weight = weights[l]

        #print "!!!!!!!!!!!!", weight.shape, weight
        print("# %s/fc%d #"%(tf.get_default_graph().get_name_scope(), l))
        dout = tf.layers.dense(dout, shapes[l], activation=None,
                               kernel_initializer=tf.initializers.constant(weight),
                               bias_initializer=tf.initializers.constant(0.1),
                               trainable=trainable,
                               name='fc%d'%l)

        units_in = shapes[l]

        if acts is None or acts[l] is None or acts[l] is '':
            continue

        act_type = acts[l]
        if act_type == 'sigmoid':
            dout = tf.nn.sigmoid(dout)
        elif act_type == 'tanh':
            dout = tf.nn.tanh(dout)
        elif act_type == 'relu':
            dout = tf.nn.relu(dout)
        elif act_type == "elu":
            dout = tf.nn.elu(dout)
        elif act_type == "softplus":
            dout = tf.nn.softplus(dout)
        elif act_type == 'prelu':
            dout = prelu(dout, trainable)
        elif act_type == 'dice':
            dout = dice(dout, is_training, trainable)
        elif act_type == 'swish':
            dout = swish(dout)
        elif act_type == 'h_swish':
            dout = h_swish(dout)
        elif act_type == 'pswish':
            dout = pswish(dout, trainable)
        elif act_type == 'mish':
            dout = mish(dout)
        elif act_type == 'gelu':
            dout = gelu(dout)
        elif act_type == 'gelu2':
            dout = gelu2(dout)
        elif not act_type:
            dout = dout
        else:
            raise RuntimeError('unknown act_type %s' % act_type)

    return dout

def multi_head_att(query, fact, fc_query_shapes, fc_fact_shapes, scope=None, indicator=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        units_query = int(query.shape[-1])
        assert units_query > 0, 'invalid query shape: %s'%query.shape

        units_fact = int(fact.shape[-1])
        assert units_fact > 0, 'invalid fact shape: %s'%fact.shape

        heads = units_query / fc_query_shapes[0]
        assert units_fact / fc_fact_shapes[0] == heads

        assert units_query % fc_query_shapes[0] == 0
        assert units_fact % fc_fact_shapes[0] == 0

        fc_querys = []
        for k in range(heads):
            with tf.variable_scope("query%d"%k):
                fc_query = fc_repeats(query, fc_query_shapes)
                fc_querys.append(fc_query)

        fc_facts = []
        for k in range(heads):
            with tf.variable_scope("fact%d"%k):
                fc_fact = fc_repeats(fact, fc_fact_shapes)
                if indicator is not None:
                    fc_fact = tf.gather(fc_fact, indices=indicator, axis=0)
                fc_facts.append(fc_fact)

        outs = []
        for k in range(3):
            dot = tf.matmul(fc_facts[k], fc_querys[k], transpose_b=True)
            alphas = tf.nn.softmax(dot, name='alphas', axis=1) + 0.0000001
            out = fc_facts[k] * alphas
            outs.append(out)

        return tf.reduce_sum(tf.concat(outs, axis=2), axis=1)

def gru(data, num_units, use_mx=False, scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print("# %s #"%tf.get_default_graph().get_name_scope())
        kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        #kernel_initializer=tf.initializers.orthogonal(gain=1.0, dtype=tf.float32)
        #kernel_initializer=tf.initializers.constant(0.1)

        if use_mx:
          from mx_rnn_cell_impl import MxGRUCell
          cell = MxGRUCell(num_units=num_units,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=tf.initializers.constant(0.0))
        else:
          cell = tf.nn.rnn_cell.GRUCell(num_units=num_units,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=tf.initializers.constant(0.0))

        outputs, last_states = tf.nn.dynamic_rnn(cell, inputs=data, dtype=tf.float32, sequence_length=None)
        mask, length = mask_to_length(data)
        outputs = outputs * mask
        return outputs, mask, length

def neg_loss_logits(data, seq, mask, is_clk, shapes, acts=None, weights=None):
    # (BATCHSIZE, GRU_STEPS-1, K*D*2)
    din = tf.concat([seq[:,1:,:], data[:,:-1,:]], axis=2)
    # (BATCHSIZE, GRU_STEPS-1, 2)
    dout = fc_repeats(din, shapes, acts, weights)
    mask = mask[:,1:,:]

    shape=[tf.shape(dout)[0], tf.shape(dout)[1], 1]

    y = tf.ones(shape) if is_clk else tf.zeros(shape)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=dout) * tf.squeeze(mask, 2)

    steps = tf.maximum(tf.reduce_sum(mask, axis=1), 1)
    return tf.reduce_sum(loss, axis=1, keepdims=True) / steps

def neg_loss_softmax(data, seq, mask, is_clk, shapes, acts=None, weights=None):
    # (BATCHSIZE, GRU_STEPS-1, K*D*2)
    din = tf.concat([seq[:,1:,:], data[:,:-1,:]], axis=2)
    # (BATCHSIZE, GRU_STEPS-1, 2)
    dout = fc_repeats(din, shapes, acts, weights)
    mask = mask[:,1:,:]

    shape=[tf.shape(dout)[0], tf.shape(dout)[1], 1]
    zeros = tf.zeros(shape)
    ones = tf.ones(shape)

    y =  tf.concat([zeros, ones], axis=2) if is_clk else tf.concat([ones, zeros], axis=2)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=dout) * tf.squeeze(mask, 2)

    steps = tf.maximum(tf.reduce_sum(mask, axis=1), 1)
    return tf.reduce_sum(loss, axis=1, keepdims=True) / steps

def neg_loss2(data, clk_seq, nclk_seq, mask, shapes, acts=None, scope=None):
    units_in = int(data.shape[-1]) * 2
    assert units_in > 0, 'invalid input shape: %s'%data.shape

    weights = list()
    for l in range(len(shapes)):
        weight = np.random.randn(units_in, shapes[l]).astype(np.float32) / np.sqrt(units_in)
        weights.append(weight)
        units_in = shapes[l]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        loss_clk = neg_loss_softmax(data, clk_seq, mask, True, shapes, acts, weights=weights)
        loss_nclk = neg_loss_softmax(data, nclk_seq, mask, False, shapes, acts, weights=weights)

    loss = (loss_clk + loss_nclk) / 2.0
    return loss

def mask_to_length(data):
    assert len(data.shape) > 2, "shape should be (batch_size, steps, ...)"
    mask = tf.to_int32(tf.not_equal(tf.reduce_sum(data, axis=range(2, len(data.shape)), keepdims=True), 0))
    length = tf.reduce_sum(mask, [1, 2])
    return tf.to_float(mask), length

# NOTE: this loss out with reduce mean
def focal_loss_bound(labels, logits, gamma=2, alpha=0.25, pos_bound=0.97, neg_bound=0.03):
    #label = tf.slice(labels, [0,1], [-1, 1])
    #logit = tf.slice(logits, [0,1], [-1, 1])
    p = tf.nn.sigmoid(logits)
    loss = -tf.reduce_mean(
        labels * alpha * tf.pow(tf.abs(pos_bound - p), gamma) * tf.log(p + 1e-6)
        + (1 - labels) * (1 - alpha) * tf.pow(tf.abs(p - neg_bound), gamma) * tf.log(1 - p + 1e-6), axis=0)
    #loss = -tf.reduce_mean(label * tf.log(logit + 1e-6) + (1-label) * (1- logit + 1e-6))
    return loss

def focal_loss(labels, logits, gamma=2, alpha=0.25):
    p =  tf.nn.sigmoid(logits)
    loss = -(labels * alpha * tf.pow(1 - p, gamma) * tf.log(p + 1e-6)
             + (1 - labels) * (1 - alpha) * tf.pow(p, gamma) * tf.log(1 - p + 1e-6))
    return loss

def focal_loss_p(labels, prob, gamma=2, alpha=0.25):
    p = tf.minimum(tf.nn.relu(prob), 1.0)
    loss = -(labels * alpha * tf.pow(1 - p, gamma) * tf.log(p + 1e-6)
             + (1 - labels) * (1 - alpha) * tf.pow(p, gamma) * tf.log(1 - p + 1e-6))
    return loss

def huber_loss(labels, logits, delta = 0.25):
    residual = tf.abs(logits - labels)
    large_loss = 0.5 * tf.square(residual)
    small_loss = delta * residual - 0.5 * tf.square(delta)
    cond = tf.less(residual, delta)
    return tf.where(cond, large_loss, small_loss)


def bias_loss(labels, logits, gamma=2, alpha=0.25, pos_bound=0.97, neg_bound=0.03):
    #label = tf.slice(labels, [0,1], [-1, 1])
    #logit = tf.slice(logits, [0,1], [-1, 1])
    N=95
    N_f = 5
    C=2
    loss = -tf.reduce_mean(
        labels  * tf.log(logits)+(N/N_f)*(C-1)*tf.log(1-logits), axis=0)
    #loss = -tf.reduce_mean(label * tf.log(logit + 1e-6) + (1-label) * (1- logit + 1e-6))
    return loss

# def get_activate_func(act_type):
#

def fc_optimal_bias(data, units, act_type, is_training, use_bn=False,N=None,N_f=None, scope=None):
    units_in = int(data.shape[-1])
    assert units_in > 0, 'invalid input shape: %s'%data.shape

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # weight_value = np.random.randn(units, units_in).astype(
        #     np.float32
        # ) / np.sqrt(units_in)
        # #print scope, "############", weight_value.shape, weight_value
        # print("# %s/fc #"%tf.get_default_graph().get_name_scope())
        # weight_value = np.transpose(weight_value)
        bias_b = -math.log(float(N-N_f)/float(N_f))
        dout = tf.layers.dense(data, units, activation=None,
                               kernel_initializer=tf.glorot_normal_initializer(),
                               bias_initializer=tf.initializers.constant(bias_b), name='fc')

        if use_bn and act_type != 'dice':
            dout = bn(dout, is_training, scope='bn')

        if act_type == 'sigmoid':
            out = tf.nn.sigmoid(dout)
        elif act_type == 'relu':
            out = tf.nn.relu(dout)
        elif act_type == 'prelu':
            alpha = tf.get_variable(
                'prelu_alpha',
                shape=(units,),
                initializer=tf.constant_initializer(-0.25),
            )
            out = tf.maximum(0.0, dout) + alpha * tf.minimum(0.0, dout)
        elif act_type == 'dice':
            out = bn(dout, is_training=is_training, center=False, scale=False, epsilon=1e-4, momentum=0.99, scope='dice_bn')
            logits = tf.nn.sigmoid(out)   # 1 / (1 + tf.exp(-out))
            dice_gamma = tf.get_variable(
                'dice_gamma',
                shape=(1, units),
                initializer=tf.constant_initializer(-0.25)
            )
            out = tf.multiply(dice_gamma, (1.0 - logits) * dout) + logits * dout
        elif not act_type:
            out = dout
        else:
            raise RuntimeError('unknown act_type %s' % act_type)
    return out

def tf_embedding_table(bucket_size,embedding_size,col):
    embeddings = tf.get_variable(
        shape=[bucket_size, embedding_size],
        initializer=tf.truncated_normal_initializer(),
        dtype=tf.float32,
        name="deep_embedding_" + col)
    return embeddings

def focal_loss_4_euen(labels, logits, gamma=2, alpha=0.25, pos_bound=0.97, neg_bound=0.03):
    #label = tf.slice(labels, [0,1], [-1, 1])
    #logit = tf.slice(logits, [0,1], [-1, 1])
    loss = -tf.reduce_mean(
        labels * alpha * tf.pow(tf.abs(pos_bound - logits), gamma) * tf.log(tf.nn.relu(logits) + 1e-6)
        + (1 - labels) * (1 - alpha) * tf.pow(tf.abs(logits - neg_bound), gamma) * tf.log(1 - tf.clip_by_value(logits,-1,1)  + 1e-6), axis=0)
    #loss = -tf.reduce_mean(label * tf.log(logit + 1e-6) + (1-label) * (1- logit + 1e-6))
    return loss

## Definitions for dragonnet and tarnet
def binary_classification_loss(concat_true, concat_pred, reduce_mean = False):
    t_true = concat_true[:, 1]
    t_pred_logit = concat_pred[:, 2]
    # t_pred = (t_pred + 0.001) / 1.002

    losst_logits = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = t_true,
        logits = t_pred_logit
    )
    if not reduce_mean:
        losst = tf.reduce_sum(losst_logits)
    else:
        losst = tf.reduce_mean(losst_logits)

    return losst

def regression_loss(concat_true, concat_pred, reduce_mean = False):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0_logits = (1. - t_true) * tf.square(y_true - y0_pred)
    loss1_logits = t_true * tf.square(y_true - y1_pred)
    if not reduce_mean:
        loss0 = tf.reduce_sum(loss0_logits)
        loss1 = tf.reduce_sum(loss1_logits)
    else:
        loss0 = tf.reduce_mean(loss0_logits)
        loss1 = tf.reduce_mean(loss1_logits)

    return loss0 + loss1

def cross_entropy_loss(concat_true, concat_pred, reduce_mean = False):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss_y = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = y_true,
        logits = t_true * y1_pred + (1. - t_true) * y0_pred
    )
    if not reduce_mean:
        loss = tf.reduce_sum(loss_y)
    else:
        loss = tf.reduce_mean(loss_y)

    return loss

def ned_loss(concat_true, concat_pred, reduce_mean = False):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]

    losst_logits = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=t_true,
        logits=t_pred
    )

    if not reduce_mean:
        losst = tf.reduce_sum(losst_logits)
    else:
        losst = tf.reduce_mean(losst_logits)
    return losst

def dead_loss(concat_true, concat_pred, reduce_mean = False):
    return regression_loss(concat_true, concat_pred, reduce_mean)

def dragonnet_loss_binarycross(concat_true, concat_pred, reduce_mean = True):
    return regression_loss(concat_true, concat_pred, reduce_mean)\
           + binary_classification_loss(concat_true, concat_pred, reduce_mean)

def treatment_accuracy(concat_true, concat_pred):
    t_true = tf.argmax(concat_true[:, 1], axis = 1)
    t_pred = tf.argmax(concat_pred[:, 2], axis = 1)
    return tf.metrics.accuracy(t_true, t_pred)

def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))

class EpsilonLayer(tf.layers.Layer):
    """
    Custom tensorflow layer to allow epsilon to be learned during training process.
    """
    def __init__(self, name="epsilon"):
        """
        Inherits tf.layers' Layer object.
        """
        super(EpsilonLayer, self).__init__()
        self.name = name

    def build(self, input_shape):
        """
        Creates a trainable weight variable for this layer.
        """
        self.epsilon = self.add_variable(shape = [1, 1],
                                         initializer = tf.random_normal_initializer(),
                                         dtype = tf.float32,
                                         trainable = True,
                                         name = self.name)
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(tf.ones_like(inputs)[:, 0:1], self.epsilon)

def make_tarreg_loss(alpha = 1., dragonnet_loss = dragonnet_loss_binarycross):
    reduce_mean = True
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred_logit = concat_pred[:, 2]
        t_pred = tf.nn.sigmoid(t_pred_logit)

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        if reduce_mean:
            targeted_regularization = tf.reduce_mean(tf.square(y_true - y_pert))
        else:
            targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + alpha * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss

def safe_sqrt(x, lbound = 1e-10):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t):
    ''' Linear MMD '''

    Xc = (1. - t) * X
    Xt = t * X

    stepc = tf.maximum(tf.reduce_sum(1. - t, axis=0), 1)
    stept = tf.maximum(tf.reduce_sum(t, axis=0), 1)

    mean_control = tf.reduce_sum(Xc, reduction_indices=0) / stepc
    mean_treated = tf.reduce_sum(Xt, reduction_indices=0) / stept

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    Xc = (1. - t) * X
    Xt = t * X

    stepc = tf.maximum(tf.reduce_sum(1. - t, axis=0), 1)
    stept = tf.maximum(tf.reduce_sum(t, axis=0), 1)

    mean_control = tf.reduce_sum(Xc, reduction_indices=0) / stepc
    mean_treated = tf.reduce_sum(Xt, reduction_indices=0) / stept

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)
    print("delta:", delta.get_shape().as_list())
    print("eff_lam:", eff_lam.get_shape().as_list())
    print("M:", M.get_shape().as_list())
    print("M[:, 0:1]:", M[:, 0:1].get_shape().as_list())
    ''' Compute new distance matrix '''
    Mt = M
    # row = delta*tf.ones(tf.shape(M[0:1,:]))
    row = delta * tf.ones_like(M[0:1, :])
    print("row:", row.get_shape().as_list())
    # col = tf.concat(0,[delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))])
    col = tf.concat([tf.expand_dims(delta * tf.ones_like(tf.shape(M[:, 0:1]),dtype=tf.float32),-1), tf.zeros((1, 1))],axis=0)
    print("col:", col.get_shape().as_list())
    # Mt = tf.concat(0,[M,row])
    Mt = tf.concat(1,[Mt,col])

    ''' Compute marginal vectors '''
    a = tf.concat(0,[p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))])
    b = tf.concat(0,[(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))])

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def group_loss(loss_logit, is_group):
    steps = tf.maximum(tf.reduce_sum(is_group, axis=0), 1)
    return tf.reduce_sum(loss_logit, axis=0) / steps

def lift_log_loss(label, is_treat, concat_pred, use_focal = False, use_group_reduce = True):
    c_prob = tf.reshape(concat_pred[:, 0], [-1, 1])
    t_prob = tf.reshape(concat_pred[:, 1], [-1, 1])

    is_t = is_treat
    is_c = 1. - is_treat

    if use_focal:
        prob_loss = focal_loss_p(label, is_c * c_prob + is_t * t_prob)
    else:
        prob_loss = tf.losses.log_loss(labels=label,
                                       predictions=is_c * c_prob + is_t * t_prob,
                                       epsilon=1e-6, reduction=tf.losses.Reduction.NONE)

    if use_group_reduce:
        reduce_loss = (group_loss(is_t * prob_loss, is_t) + group_loss(is_c * prob_loss, is_c)) / 2.0
    else:
        reduce_loss = tf.reduce_mean(prob_loss)

    return reduce_loss

def lift_ce_loss(label, is_treat, concat_pred, use_focal = False, use_group_reduce = True):
    c_logit = tf.reshape(concat_pred[:, 0], [-1, 1])
    t_logit = tf.reshape(concat_pred[:, 1], [-1, 1])

    is_t = is_treat
    is_c = 1. - is_treat

    if use_focal:
        prob_loss = focal_loss(label, is_c * c_logit + is_t * t_logit)
    else:
        prob_loss =  tf.nn.sigmoid_cross_entropy_with_logits(labels=label,
                                                             logits=is_t * t_logit + is_c * c_logit)

    if use_group_reduce:
        reduce_loss = (group_loss(is_t * prob_loss, is_t) + group_loss(is_c * prob_loss, is_c)) / 2.0
    else:
        reduce_loss = tf.reduce_mean(prob_loss)

    return reduce_loss

def lift_mse_loss(label, is_treat, concat_pred, use_huber = False, use_group_reduce = True):
    c_pred = tf.reshape(concat_pred[:, 0], [-1, 1])
    t_pred = tf.reshape(concat_pred[:, 1], [-1, 1])

    is_t = is_treat
    is_c = 1. - is_treat

    if use_huber:
        loss = huber_loss(label, is_c * c_pred + is_t * t_pred, 0.25)
    else:
        loss = tf.square(is_c * c_pred + is_t * t_pred - label)

    if use_group_reduce:
        reduce_loss = (group_loss(is_t * loss, is_t) + group_loss(is_c * loss, is_c)) / 2.0
    else:
        reduce_loss = tf.reduce_mean(loss)

    return reduce_loss
