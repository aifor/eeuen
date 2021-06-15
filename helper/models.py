# -*- coding: utf-8 -*-

import xdl
import tensorflow as tf
import helper.utils as utils

class EUEN(object):
    def __init__(self, input, is_treat, label, hc_dim, hu_dim, is_self, l2_reg, act_type = "elu", is_training = True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.hc_dim = hc_dim
        self.hu_dim = hu_dim
        self.is_self = is_self
        self.l2_reg = l2_reg
        self.act_type = act_type
        self.is_training = is_training

    # control net
    def ControlNet(self):
        c_fc1 = utils.fc(self.input, self.hc_dim, act_type=self.act_type, is_training=self.is_training, scope='c_fc1')
        c_fc2 = utils.fc(c_fc1, self.hc_dim / 2, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc2')
        c_fc3 = utils.fc(c_fc2, self.hc_dim / 4, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc3')
        c_last = c_fc3
        if self.is_self:
            c_fc4 = utils.fc(c_fc3, self.hc_dim / 8, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc4')
            c_last = c_fc4

        c_logit = utils.fc(c_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='c_logit')
        c_tau = utils.fc(c_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='c_tau')
        c_prob = tf.nn.sigmoid(c_logit)
        return c_logit, c_prob, c_tau

    # uplift net
    def UpliftNet(self):
        u_fc1 = utils.fc(self.input, self.hu_dim, act_type=self.act_type, is_training=self.is_training, scope='u_fc1')
        u_fc2 = utils.fc(u_fc1, self.hu_dim / 2, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc2')
        u_fc3 = utils.fc(u_fc2, self.hu_dim / 4, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc3')
        u_last = u_fc3
        if self.is_self:
            u_fc4 = utils.fc(u_fc3, self.hu_dim / 8, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc4')
            u_last = u_fc4
        t_logit = utils.fc(u_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='t_logit')
        u_tau = utils.fc(u_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='u_tau')

        t_prob = tf.nn.sigmoid(t_logit)
        return t_logit, t_prob, u_tau

    def EUENFit(self):
        c_logit, c_prob, c_tau = self.ControlNet()
        t_logit, t_prob, u_tau = self.UpliftNet()

        # regression
        c_logit_fix = tf.stop_gradient(c_logit)

        uc = c_logit
        ut = c_logit_fix + u_tau
        # ut = c_logit + u_tau

        # response loss
        resp_sq_loss = utils.lift_mse_loss(self.label, self.is_treat, tf.concat([uc, ut], 1),
                                           use_huber=False, use_group_reduce=False)
        total_loss = resp_sq_loss

        u = (1 - self.is_treat) * uc + self.is_treat * ut
        mse, mse_op = tf.contrib.metrics.streaming_mean_squared_error(u, self.label[:, :1])

        if self.is_training:
            return total_loss, u_tau, mse, mse_op
        else:
            xdl.trace_tf_tensor('uplift_score', u_tau)
            print("is_treat shape:", self.is_treat.get_shape().as_list())
            ate = tf.reduce_mean(u_tau)
            print("ate:", ate.get_shape().as_list())
            return total_loss, ate, mse, mse_op

class EEUEN(object):
    def __init__(self, input, is_treat, is_exp, label, hc_dim, hu_dim, he_dim,
                 is_self, l2_reg, act_type="elu", is_training=True):
        self.input = input
        self.is_treat = is_treat
        self.is_exp = is_exp
        self.label = label
        self.hc_dim = hc_dim
        self.hu_dim = hu_dim
        self.he_dim = he_dim
        self.is_self = is_self
        self.l2_reg = l2_reg
        self.act_type = act_type
        self.is_training = is_training

    # control net
    def ControlNet(self):
        c_fc1 = utils.fc(self.input, self.hc_dim, act_type=self.act_type, is_training=self.is_training, scope='c_fc1')
        c_fc2 = utils.fc(c_fc1, self.hc_dim / 2, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc2')
        c_fc3 = utils.fc(c_fc2, self.hc_dim / 4, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc3')
        c_last = c_fc3
        if self.is_self:
            c_fc4 = utils.fc(c_fc3, self.hc_dim / 8, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='c_fc4')
            c_last = c_fc4

        c_logit = utils.fc(c_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='c_logit')
        c_tau = utils.fc(c_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='c_tau')
        c_prob = tf.nn.sigmoid(c_logit)
        return c_logit, c_prob, c_tau

    # treat exposure net
    def TreatExpNet(self):
        e_fc1 = utils.fc(self.input, self.he_dim, act_type=self.act_type, is_training=self.is_training, scope='e_fc1')
        e_fc2 = utils.fc(e_fc1, self.he_dim / 2, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='e_fc2')
        e_fc3 = utils.fc(e_fc2, self.he_dim / 4, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='e_fc3')
        e_last = e_fc3
        if self.is_self:
            e_fc4 = utils.fc(e_fc3, self.he_dim / 8, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='e_fc4')
            e_last = e_fc4
        e_logit = utils.fc(e_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='e_logit')
        e_prob = tf.nn.sigmoid(e_logit)
        return e_logit, e_prob

    # uplift net
    def UpliftNet(self):
        u_fc1 = utils.fc(self.input, self.hu_dim, act_type=self.act_type, is_training=self.is_training, scope='u_fc1')
        u_fc2 = utils.fc(u_fc1, self.hu_dim / 2, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc2')
        u_fc3 = utils.fc(u_fc2, self.hu_dim / 4, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc3')
        u_last = u_fc3
        if self.is_self:
            u_fc4 = utils.fc(u_fc3, self.hu_dim / 8, act_type=self.act_type, l2_reg=self.l2_reg, is_training=self.is_training, scope='u_fc4')
            u_last = u_fc4
        t_logit = utils.fc(u_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='t_logit')
        u_tau = utils.fc(u_last, 1, act_type='', l2_reg=self.l2_reg, is_training=self.is_training, scope='u_tau')

        t_prob = tf.nn.sigmoid(t_logit)
        return t_logit, t_prob, u_tau

    def ExpPreFit(self, model_scope):
        with tf.variable_scope('assign_net') as scope:
            e_logit, e_prob = self.TreatExpNet()

        exp_loss = tf.reduce_mean(
            self.is_treat * tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_exp, logits=e_logit))
        return exp_loss, e_prob

    def EEUENFit(self, model_scope):
        c_logit, c_prob, c_tau = self.ControlNet()
        t_logit, t_prob, u_tau = self.UpliftNet()
        with tf.variable_scope('assign_net') as scope:
            e_logit, e_prob = self.TreatExpNet()

        c_logit_fix = tf.stop_gradient(c_logit)

        uc = c_logit
        ut = c_logit_fix + e_prob * u_tau

        # response loss
        hat_loss = utils.lift_mse_loss(self.label, self.is_treat, tf.concat([uc, ut], 1),
                                       use_huber=False, use_group_reduce=False)

        # exposure loss
        exp_loss_logit = self.is_treat * tf.losses.log_loss(labels=self.is_exp,
                                                            predictions=e_prob,
                                                            epsilon=1e-6, reduction=tf.losses.Reduction.NONE)
        # exp_loss = utils.group_loss(exp_loss_logit, self.is_treat)
        exp_loss = tf.reduce_mean(exp_loss_logit)

        total_loss = hat_loss + exp_loss

        u = (1 - self.is_treat) * uc + self.is_treat * ut
        mse, mse_op = tf.contrib.metrics.streaming_mean_squared_error(u, self.label[:, :1])
        if self.is_training:
            # xdl.trace_tf_tensor('score', tf.sigmoid(u_logits))
            return total_loss, u_tau, mse, mse_op
        else:
            xdl.trace_tf_tensor('multi_uplift_score', e_prob * u_tau, scope=model_scope)
            xdl.trace_tf_tensor('uplift_score', u_tau, scope=model_scope)
            xdl.trace_tf_tensor('e_prob', e_prob, scope=model_scope)
            print("is_treat shape:", self.is_treat.get_shape().as_list())
            ate = tf.reduce_mean(u_tau)
            print("ate:", ate.get_shape().as_list())

            return total_loss, ate, mse, mse_op

class CEVAE(object):
    def __init__(self, input, is_treat, label, h_dim, x_repr_dim, z_repr_dim,
                 rep_norm, is_self, act_type="elu", is_training=True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.h_dim = h_dim
        self.x_repr_dim = x_repr_dim
        self.z_repr_dim = z_repr_dim
        self.rep_norm = rep_norm
        self.is_self = is_self
        self.act_type = act_type
        self.is_training = is_training

    def RepX4Emb(self):
        """ R(X) Base Network to representation inputs
        """
        R_fc1 = utils.fc(self.input, self.h_dim, self.act_type, self.is_training, init_w_type="xavier", scope="r_fc1")
        X_repr = utils.fc(R_fc1, self.x_repr_dim, self.act_type, self.is_training, init_w_type="xavier", scope="x_repr")
        if self.rep_norm:
            X_repr_norm = X_repr / utils.safe_sqrt(tf.reduce_sum(tf.square(X_repr), axis=1, keep_dims=True))
        else:
            X_repr_norm = 1.0 * X_repr

        return X_repr_norm

    def RepX(self):
        """ R(X) Base Network to representation inputs
        """
        if self.rep_norm:
            X_repr_norm = self.input / utils.safe_sqrt(tf.reduce_sum(tf.square(self.input), axis=1, keep_dims=True))
        else:
            X_repr_norm = self.input

        return X_repr_norm

    def P_X_Z(self, z):
        """ P(X|Z) Network to representation X
        Args:
            - z: latent variables
        Returns:
            - X_Z_repr: representation of X for input Z
        """

        X_Z_fc1 = utils.fc(z, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="x_z_fc1")
        X_Z_fc2 = utils.fc(X_Z_fc1, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="x_z_fc2")
        X_Z_fc3 = utils.fc(X_Z_fc2, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="x_z_fc3")

        X_Z_mu = utils.fc(X_Z_fc3, self.x_repr_dim, act_type="", is_training=self.is_training, init_w_type="xavier", scope="x_z_mu")
        X_Z_sigma = utils.fc(X_Z_fc3, self.x_repr_dim, act_type="softplus", is_training=self.is_training, init_w_type="xavier",
                             scope="x_z_sigma")
        if self.is_self:
            sigma = X_Z_sigma + 1e-4
        else:
            sigma = tf.maximum(X_Z_sigma, 0.4)
        X_Z_repr = tf.distributions.Normal(X_Z_mu, sigma, validate_args=True, name="x_z_repr")

        return X_Z_repr

    def P_T_Z(self, z):
        """ P(T|Z) Network to predict T
        Args:
            - z: latent variables
        Returns:
            - T_Z_bin: binary representation of T for input Z
        """
        T_Z_fc1 = utils.fc(z, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="t_z_fc1")
        T_Z_fc2 = utils.fc(T_Z_fc1, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="t_z_fc2")
        T_Z_logit = utils.fc(T_Z_fc2, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="t_z_logit")
        T_Z_p = tf.nn.sigmoid(T_Z_logit)

        # for treat discrete out
        T_Z_bin = tf.distributions.Bernoulli(probs=tf.maximum(T_Z_p, 1e-4), dtype=tf.float32, validate_args=True,
                                             name="t_z_bin")

        return T_Z_bin, T_Z_p, T_Z_logit

    def P_Y_ZT(self, z, t):
        """ P(Y|Z, T) Network to predict Y
        Args:
            - z: latent variables
            - t: treat variables
        Returns:
            - y_prob: probability of Y for input Z and T
        """
        X_t0_fc1 = utils.fc(z, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t0_fc1")
        X_t0_fc2 = utils.fc(X_t0_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t0_fc2")
        X_t0_fc3 = utils.fc(X_t0_fc2, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t0_fc3")
        X_t0_last = X_t0_fc3
        if self.is_self:
            X_t0_fc4 = utils.fc(X_t0_fc3, int(self.h_dim / 8), act_type=self.act_type, is_training=self.is_training, init_w_type = "xavier", scope="x_z_t0_fc4")
            X_t0_last = X_t0_fc4
        mu_t0 = utils.fc(X_t0_last, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="x_z_mu_t0")

        X_t1_fc1 = utils.fc(z, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t1_fc1")
        X_t1_fc2 = utils.fc(X_t1_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t1_fc2")
        X_t1_fc3 = utils.fc(X_t1_fc2, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                            scope="x_z_t1_fc3")
        X_t1_last = X_t1_fc3
        if self.is_self:
            X_t1_fc4 = utils.fc(X_t1_fc3, int(self.h_dim / 8), act_type=self.act_type, is_training=self.is_training, init_w_type = "xavier", scope="x_z_t1_fc4")
            X_t1_last = X_t1_fc4
        mu_t1 = utils.fc(X_t1_last, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="x_z_mu_t1")

        y_prob = tf.distributions.Normal((1 - t) * mu_t0 + t * mu_t1, 1., validate_args=True, name="y_zt_prob")

        return y_prob

    ## Inference Model / Encoder
    def Q_T_X(self, x):
        """ P(T|X) Network to representation T
        Args:
            - x: represent variables
        Returns:
            - T_X_bin: binary representation of T for input X
        """

        T_X_fc1 = utils.fc(x, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="t_x_fc1")
        T_X_fc2 = utils.fc(T_X_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                           scope="t_x_fc2")
        T_X_logit = utils.fc(T_X_fc2, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="t_x_logit")
        T_X_p = tf.nn.sigmoid(T_X_logit)

        # for treat discrete out
        T_X_bin = tf.distributions.Bernoulli(probs=tf.maximum(T_X_p, 1e-4), dtype=tf.float32, validate_args=True,
                                             name="t_x_bin")

        return T_X_bin, T_X_logit, T_X_p

    def Q_Y_XT(self, x, t):
        """ P(Y|X, T) Network to predict Y
        Args:
            - x: represent variables
            - t: treat variables
        Returns:
            - y_prob: probability of Y for input X and T
        """
        X_fc1 = utils.fc(x, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier", scope="x_fc1")
        X_fc2 = utils.fc(X_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                         scope="x_fc2")
        X_fc3 = utils.fc(X_fc2, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                         scope="x_fc3")
        X_last = X_fc3
        if self.is_self:
            X_fc4 = utils.fc(X_fc3, int(self.h_dim / 8), act_type=self.act_type, is_training=self.is_training, init_w_type = "xavier", scope="x_fc4")
            X_last = X_fc4

        mu_t0 = utils.fc(X_last, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="x_mu_t0")
        mu_t1 = utils.fc(X_last, 1, act_type="", is_training=self.is_training, init_w_type="xavier", scope="x_mu_t1")

        # set mu according to t, sigma set to 1
        y_prob = tf.distributions.Normal((1. - t) * mu_t0 + t * mu_t1, 1., validate_args=True, name="y_xt_prob")

        return y_prob

    def Q_Z_TYX(self, x, y, t):
        """ P(Z|T, Y, X) Network to predict Z latent representation
        Args:
            - x: represent variables
            - y: label variables
            - t: treat  variables
        Returns:
            - z_latent: latent representation of Z for input T, Y, X
        """
        xy = tf.concat(values=[x, y], axis=1)

        XY_fc1 = utils.fc(xy, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier", scope="xy_fc1")
        XY_fc2 = utils.fc(XY_fc1, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                          scope="xy_fc2")
        XY_fc3 = utils.fc(XY_fc2, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="xavier",
                          scope="xy_fc3")
        XY_last = XY_fc3
        if self.is_self:
            XY_fc4 = utils.fc(XY_fc3, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type = "xavier", scope="xy_fc4")
            XY_last = XY_fc4

        mu_t0 = utils.fc(XY_last, self.z_repr_dim, act_type="", is_training=self.is_training, init_w_type="xavier",
                         scope="xy_mu_t0")
        mu_t1 = utils.fc(XY_last, self.z_repr_dim, act_type="", is_training=self.is_training, init_w_type="xavier",
                         scope="xy_mu_t1")

        sigma_t0 = utils.fc(XY_last, self.z_repr_dim, act_type="softplus", is_training=self.is_training, init_w_type="xavier",
                            scope="xy_sigma_t0")
        sigma_t1 = utils.fc(XY_last, self.z_repr_dim, act_type="softplus", is_training=self.is_training, init_w_type="xavier",
                            scope="xy_sigma_t1")

        if self.is_self:
            sigma = (1. - t) * sigma_t0 + t * sigma_t1 + 1e-4
        else:
            sigma = tf.maximum((1. - t) * sigma_t0 + t * sigma_t1, 0.4)
        Z_latent = tf.distributions.Normal((1. - t) * mu_t0 + t * mu_t1, sigma, validate_args=True, name="z_xyt_latent")
        return Z_latent

    def ZPrior(self, Z_latent):
        return tf.distributions.Normal(tf.zeros_like(Z_latent, dtype=tf.float32),
                                       tf.ones_like(Z_latent, dtype=tf.float32),
                                       validate_args=True)
        # return tf.distributions.Bernoulli(probs=0.5 * tf.ones_like(Z_latent, dtype=tf.float32), dtype=tf.float32, validate_args=True)

    ## CEVAE model
    def FixXFit(self):
        ## Structure
        with tf.variable_scope('rep_net', reuse=True) as scope:
            X_repr, logit = self.RepX4Emb()

        loss_logit = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit)
        return tf.reduce_mean(loss_logit)

    def ZInitFit(self):
        ## Structure
        # get X representation
        if self.is_self:
            with tf.variable_scope('rep_net', reuse=True) as scope:
                X_repr, _ = self.RepX4Emb()
            X_repr = tf.stop_gradient(X_repr)
        else:
            # use only for sparse data with embedding
            X_repr = self.RepX()

        # Predict y and t based x
        # T_X_bin = self.Q_T_X(X_repr)
        # Y_XT_prob = self.Q_Y_XT(X_repr, self.is_treat)

        # get prior laten z
        Z_latent = self.Q_Z_TYX(X_repr, self.label, self.is_treat)
        sample_Z = Z_latent.sample()
        # mean_Z = Z_latent.mean()
        Z_latent_prior = self.ZPrior(sample_Z)
        # loss_Z = tf.reduce_sum(Z_latent.log_prob(Z_latent.mean()) - Z_latent_prior.log_prob(Z_latent.mean()), axis=1)
        loss_Z = tf.reduce_sum(tf.distributions.kl_divergence(Z_latent, Z_latent_prior, allow_nan_stats=True), axis=1)
        # loss_Z = tf.reduce_mean(tf.distributions.kl_divergence(Z_latent, Z_latent_prior, allow_nan_stats=True), axis=1)
        return tf.reduce_mean(loss_Z)

    def CEVAEFit(self):
        ## Structure
        # get X representation
        if self.is_self:
            with tf.variable_scope('rep_net', reuse=True) as scope:
                X_repr, _ = self.RepX4Emb()
            X_repr = tf.stop_gradient(X_repr)
        else:
            # use only for sparse data with embedding
            X_repr = self.RepX()

        # Predict y and t based x
        T_X_bin, tx_logit, tx_p = self.Q_T_X(X_repr)

        Y_XT_prob = self.Q_Y_XT(X_repr, self.is_treat)

        # get prior laten z
        Z_latent = self.Q_Z_TYX(X_repr, self.label, self.is_treat)
        sample_Z = Z_latent.sample()
        # mean_Z = Z_latent.mean()
        Z_latent_prior = self.ZPrior(sample_Z)

        # get x, y and t desc by giving z
        X_Z_repr = self.P_X_Z(sample_Z)
        T_Z_bin, zp_last, zp_logit = self.P_T_Z(sample_Z)
        Y_ZT_prob = self.P_Y_ZT(sample_Z, self.is_treat)

        loss_X = tf.reduce_sum(X_Z_repr.log_prob(X_repr), axis=1)
        # loss_X = tf.reduce_mean(X_Z_repr.log_prob(X_repr), axis=1)
        loss_X = tf.where(tf.is_nan(loss_X), -1e-8 * tf.ones_like(loss_X), loss_X)

        loss_Y = tf.reduce_sum(Y_ZT_prob.log_prob(self.label), axis=1)
        loss_Y = tf.where(tf.is_nan(loss_Y), -1e-8 * tf.ones_like(loss_Y), loss_Y)

        loss_T = tf.reduce_sum(T_Z_bin.log_prob(self.is_treat), axis=1)
        loss_T = tf.where(tf.is_nan(loss_T), -1e-8 * tf.ones_like(loss_T), loss_T)

        # loss_Z = tf.reduce_mean(-tf.distributions.kl_divergence(Z_latent, Z_latent_prior, allow_nan_stats=True), axis=1)
        loss_Z = tf.reduce_sum(-tf.distributions.kl_divergence(Z_latent, Z_latent_prior, allow_nan_stats=True), axis=1)
        loss_Z = tf.where(tf.is_nan(loss_Z), -1e-8 * tf.ones_like(loss_Z), loss_Z)

        loss_PT = tf.reduce_sum(T_X_bin.log_prob(self.is_treat), axis=1)
        loss_PT = tf.where(tf.is_nan(loss_PT), -1e-8 * tf.ones_like(loss_PT), loss_PT)

        loss_PY = tf.reduce_sum(Y_XT_prob.log_prob(self.label), axis=1)
        loss_PY = tf.where(tf.is_nan(loss_PY), -1e-8 * tf.ones_like(loss_PY), loss_PY)

        loss_elbo = loss_X + loss_Y + loss_T + loss_Z
        loss_pred = loss_PT + loss_PY

        loss = tf.reduce_mean(-loss_elbo - loss_pred)

        mean_tx_p = tf.reduce_mean(tx_p)
        mean_zp_last = tf.reduce_mean(zp_last)
        mean_X_repr = tf.reduce_mean(X_repr)
        mean_z_sample = tf.reduce_mean(sample_Z)
        mean_z_prior = tf.reduce_mean(Z_latent_prior.sample())

        mean_loss_x = tf.reduce_mean(loss_X)
        mean_loss_y = tf.reduce_mean(loss_Y)
        mean_loss_t = tf.reduce_mean(loss_T)
        mean_loss_pt = tf.reduce_mean(loss_PT)
        mean_loss_py = tf.reduce_mean(loss_PY)
        mean_loss_z = tf.reduce_mean(loss_Z)

        return loss, mean_z_sample \
            , mean_tx_p, mean_zp_last \
            , mean_X_repr, mean_z_prior \
            , mean_loss_x, mean_loss_y \
            , mean_loss_t, mean_loss_pt \
            , mean_loss_py, mean_loss_z

    def CEVAEPred(self):
        ## Structure
        # get X representation
        if self.is_self:
            with tf.variable_scope('rep_net', reuse=True) as scope:
                X_repr, _ = self.RepX4Emb()
            X_repr = tf.stop_gradient(X_repr)
        else:
            # use only for sparse data with embedding
            X_repr = self.RepX()

        # Predict y and t based x
        T_X_bin, T_X_logit, T_X_p = self.Q_T_X(X_repr)
        sample_T = tf.cast(T_X_bin.sample(), dtype=tf.float32)
        # sample_T = T_X_bin.sample()
        Y_XT_prob = self.Q_Y_XT(X_repr, sample_T)
        sample_Y = Y_XT_prob.sample()

        # get prior laten z
        Z_latent = self.Q_Z_TYX(X_repr, sample_Y, sample_T)
        sample_Z = Z_latent.sample()
        # Z_latent_prior = latent_prior(Z_latent)
        print("batch_size0:", sample_Z.get_shape().as_list()[0])
        print("batch_size-1:", sample_Z.get_shape().as_list()[-1])
        y0 = self.P_Y_ZT(sample_Z, tf.zeros_like(sample_T))
        y1 = self.P_Y_ZT(sample_Z, tf.ones_like(sample_T))

        loss = tf.reduce_mean(sample_Z - sample_Y)
        uplift_score = y1.mean() - y0.mean()
        xdl.trace_tf_tensor('uplift_score', uplift_score)
        ate = tf.reduce_mean(uplift_score)

        return loss, ate

class CFRNET(object):
    def __init__(self, input, is_treat, label, h_dim, rep_norm,
                 is_self, l2_reg, act_type="elu", is_training=True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.h_dim = h_dim
        self.rep_norm = rep_norm
        self.is_self = is_self
        self.l2_reg = l2_reg
        self.act_type = act_type
        self.is_training = is_training

    # Define counterfactual representation network: balancing neural network
    def MakeBNN(self, is_treat):
        """ Neural net predictive model. The BNN has one heads

        Args:
        Returns:
            - B_logit: estimated potential outcomes
            - B_rep_norm: treat representation
        """
        ## Multistage One Define
        # in representation parts
        B_ifc1 = utils.fc(self.input, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="b_ifc1")
        B_ifc2 = utils.fc(B_ifc1, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                          drop_rate=None, init_w_type='norm', scope="b_ifc2")
        # representation for treat
        if self.rep_norm:
            B_rep_norm = B_ifc2 / utils.safe_sqrt(tf.reduce_sum(tf.square(B_ifc2), axis=1, keep_dims=True))
        else:
            B_rep_norm = 1.0 * B_ifc2

        B_rep = tf.concat(values=[B_rep_norm, is_treat], axis=1, name='b_rep')

        ## out representation parts
        y_ofc1 = utils.fc(B_rep, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y_ofc1")
        y_ofc2 = utils.fc(y_ofc1, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y_ofc2")
        B_logit = utils.fc(y_ofc2, 1, act_type="", is_training=self.is_training,
                           l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y_logit")

        return B_logit, B_rep_norm

    def MakeCFR(self):
        """ Neural net predictive model. The BNN has two heads

        Args:
        Returns:
            - B_logit: estimated potential outcomes
            - B_rep_norm: treat representation
        """
        ## Multistage One Define
        # in representation parts
        B_ifc1 = utils.fc(self.input, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="b_ifc1")
        B_ifc2 = utils.fc(B_ifc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                          drop_rate=None, init_w_type='norm', scope="b_ifc2")

        # representation for treat
        if self.rep_norm:
            B_rep_norm = B_ifc2 / utils.safe_sqrt(tf.reduce_sum(tf.square(B_ifc2), axis=1, keep_dims=True))
        else:
            B_rep_norm = 1.0 * B_ifc2

        ## out representation parts
        # y0 estimate
        y0_ofc1 = utils.fc(B_rep_norm, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                           l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y0_ofc1")
        y0_last = y0_ofc1
        if self.is_self:
            y0_ofc2 = utils.fc(y0_ofc1, int(self.h_dim / 8), act_type=self.act_type, is_training=self.is_training,
                               l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y0_ofc2")
            y0_last = y0_ofc2
        y0_logit = utils.fc(y0_last, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y0_logit")

        # y1 estimate
        y1_ofc1 = utils.fc(B_rep_norm, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                           l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y1_ofc1")
        y1_last = y1_ofc1
        if self.is_self:
            y1_ofc2 = utils.fc(y1_ofc1, int(self.h_dim / 8), act_type=self.act_type, is_training=self.is_training,
                               l2_reg=self.l2_reg, use_bn=False, drop_rate=None, init_w_type='norm', scope="y1_ofc2")
            y1_last = y1_ofc2
        y1_logit = utils.fc(y1_last, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y1_logit")

        B_logit = tf.concat(values=[y0_logit, y1_logit], axis=1)

        return B_logit, B_rep_norm

    ## BNN model
    def BNNFit(self, total_treat_prob):
        ## Structure
        # predicts
        B_logit, B_rep_norm = self.MakeBNN()

        # total loss
        alpha = 1e-4
        pred_loss = tf.reduce_mean(tf.square(self.label - B_logit))
        # pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label = self.label, logits = B_logit))
        imb_dist = utils.mmd2_lin(B_rep_norm, self.is_treat, total_treat_prob)
        imb_loss = alpha * utils.safe_sqrt(imb_dist)
        # imb_dist, imb_mat = utils.wasserstein(B_rep_norm, self.is_treat, total_treat_prob, 10, 10, False, True)
        # imb_loss = 1e-4 * imb_dist

        return pred_loss + imb_loss, B_logit

    def CFRFit(self, total_treat_prob):
        ## Structure
        # predicts
        B_logit, B_rep_norm = self.MakeCFR()

        concat_true = tf.concat(values=[self.label, self.is_treat], axis=1)
        # total loss
        alpha = 1e-4
        pred_loss = utils.regression_loss(concat_true, B_logit, True)
        # pred_loss = utils.cross_entropy_loss(concat_true, B_logit, True)
        imb_dist = utils.mmd2_lin(B_rep_norm, self.is_treat, total_treat_prob)
        imb_loss = alpha * utils.safe_sqrt(imb_dist)
        # imb_dist, imb_mat = utils.wasserstein(B_rep_norm, self.is_treat, total_treat_prob, 10, 10, False, True)
        # imb_loss = 1e-4 * imb_dist

        return pred_loss + imb_loss, B_logit


class DRAGONNET(object):
    def __init__(self, input, is_treat, label, h_dim,
                 is_self, l2_reg, alpha = 1., targeted_reg = True, act_type="elu", is_training=True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.h_dim = h_dim
        self.is_self = is_self
        self.alpha = alpha
        self.targeted_reg = targeted_reg
        self.l2_reg = l2_reg
        self.act_type = act_type
        self.is_training = is_training

    # Define dragonnet
    def MakeDragonnet(self):
        """ Neural net predictive model. The dragon has three heads

        Args:
        Returns:
            - D_logit: estimated potential outcomes
        """
        # MLP parts
        D_fc1 = utils.fc(self.input, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type="norm", scope="d_fc1")
        D_last = D_fc1
        h_dim = self.h_dim
        if self.is_self:
            D_fc2 = utils.fc(D_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type = "norm", scope="d_fc2")
            # D_fc3 = utils.fc(D_fc2, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training, init_w_type = "norm", scope="d_fc3")
            D_last = D_fc2
            h_dim = int(self.h_dim / 2)

        # propensity for treat predict
        t_logit = utils.fc(D_last, 1, act_type="", is_training=self.is_training, init_w_type="norm", scope="t_logit")

        # hypothesis
        y0_fc1 = utils.fc(D_last, int(h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y0_fc1")
        y0_fc2 = utils.fc(y0_fc1, int(h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y0_fc2")

        y1_fc1 = utils.fc(D_last, int(h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y1_fc1")
        y1_fc2 = utils.fc(y1_fc1, int(h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y1_fc2")

        # estimated outcome if treat = 0
        y0_logit = utils.fc(y0_fc2, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y0_logit")

        # estimated outcome if treat = 1
        y1_logit = utils.fc(y1_fc2, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type="norm", scope="y1_logit")

        # Epsilon layer
        with tf.variable_scope('Epsilon', reuse=tf.AUTO_REUSE) as scope:
            dl = tf.get_variable(name='epsilon', shape=[1, 1],
                                 initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                 dtype=tf.float32)
            epsilons = dl * tf.ones_like(t_logit)[:, 0:1]

        D_logit = tf.concat(values=[y0_logit, y1_logit, t_logit, epsilons], axis=1)

        return D_logit

    def DragonnetFit(self):
        ## Structure
        # Logits
        D_logit = self.MakeDragonnet()

        # Losses
        concat_true = tf.concat(values=[self.label, self.is_treat], axis=1)
        if self.targeted_reg:
            loss = utils.make_tarreg_loss(self.alpha, dragonnet_loss=utils.dragonnet_loss_binarycross)
        else:
            loss = utils.dragonnet_loss_binarycross
        D_loss = loss(concat_true, D_logit)

        return D_loss, D_logit


class GANITE(object):
    def __init__(self, input, is_treat, label, h_dim,
                 is_self, act_type="relu", is_training=True, trainable=True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.h_dim = h_dim
        self.is_self = is_self
        self.act_type = act_type
        self.is_training = is_training
        self.trainable = trainable

    ## Define Generator, Discriminator and Inference networks
    # Define generator
    def Generator(self):
        """ Generator function

        Args:
        Returns:
            - G_logit: estimated potential outcomes
        """
        # inputs were consist of feature, treatments and label
        inputs = tf.concat(values=[self.input, self.is_treat, self.label], axis=1)
        G_fc1 = utils.fc(inputs, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="g_fc1")
        G_fc2 = utils.fc(G_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="g_fc2")

        # estimated outcome if treat = 0
        G_fc31 = utils.fc(G_fc2, int(self.h_dim / 3), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="g_fc31")
        G_fc32 = utils.fc(G_fc31, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="g_fc32")
        G_logit1 = utils.fc(G_fc32, 1, act_type="", is_training=self.is_training,
                            trainable=self.trainable, init_w_type="xavier", scope="g_logit1")

        # estimated outcome if treat = 1
        G_fc41 = utils.fc(G_fc2, int(self.h_dim / 3), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="g_fc41")
        G_fc42 = utils.fc(G_fc41, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="g_fc42")
        G_logit2 = utils.fc(G_fc42, 1, act_type="", is_training=self.is_training,
                            trainable=self.trainable, init_w_type="xavier", scope="g_logit2")

        G_logit = tf.concat(values=[G_logit1, G_logit2], axis=1)
        G_tilde = tf.nn.sigmoid(G_logit)

        return G_logit, G_tilde

    # Define Discriminator
    def Discriminator(self, hat_label):
        """ Discriminator function

        Args:
            - hat_label: estimated counterfactuals
        Returns:
            - D_logit: estimated potential outcomes
        """

        # Concatenate factual & counterfactual outcomes
        # if treat = 0
        i0 = (1. - self.is_treat) * self.label + self.is_treat * tf.reshape(hat_label[:, 0], [-1, 1])
        # if treat = 1
        i1 = self.is_treat * self.label + (1. - self.is_treat) * tf.reshape(hat_label[:, 1], [-1, 1])

        inputs = tf.concat(values=[self.input, i0, i1], axis=1)
        D_fc1 = utils.fc(inputs, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="d_fc1")
        D_fc2 = utils.fc(D_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="d_fc2")
        D_fc3 = utils.fc(D_fc2, int(self.h_dim / 3), act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="d_fc3")
        D_fc4 = utils.fc(D_fc3, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="d_fc4")
        D_logit = utils.fc(D_fc4, 1, act_type="", is_training=self.is_training,
                           trainable=self.trainable, init_w_type="xavier", scope="d_logit")

        return D_logit

    # Define Inference
    def Inference(self):
        """ Inference function

        Args:
        Returns:
            - I_logit: estimated potential outcomes
        """

        I_fc1 = utils.fc(self.input, self.h_dim, act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="i_fc1")
        I_fc2 = utils.fc(I_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                         trainable=self.trainable, init_w_type="xavier", scope="i_fc2")

        # Estimated outcome if t = 0
        I_fc31 = utils.fc(I_fc2, int(self.h_dim / 3), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="i_fc31")
        I_fc32 = utils.fc(I_fc31, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="i_fc32")
        I_logit1 = utils.fc(I_fc32, 1, act_type="", is_training=self.is_training,
                            trainable=self.trainable, init_w_type="xavier", scope="i_logit1")

        # Estimated outcome if t = 1
        I_fc41 = utils.fc(I_fc2, int(self.h_dim / 3), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="i_fc41")
        I_fc42 = utils.fc(I_fc41, int(self.h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          trainable=self.trainable, init_w_type="xavier", scope="i_fc42")
        I_logit2 = utils.fc(I_fc42, 1, act_type="", is_training=self.is_training,
                            trainable=self.trainable, init_w_type="xavier", scope="i_logit2")

        I_logit = tf.concat(values=[I_logit1, I_logit2], axis=1)
        return I_logit

    def DFit(self):
        with tf.variable_scope('model') as scope:
            # Generator
            with tf.variable_scope('Generator', reuse=True) as scope:
                Y_tilde_logit, Y_tilde = self.Generator()

            # Discriminator
            with tf.variable_scope('Discriminator', reuse=True) as scope:
                D_logit = self.Discriminator(Y_tilde)

            D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_treat, logits=D_logit))
        return D_loss

    def GFit(self):
        with tf.variable_scope('model') as scope:
            # Generator
            with tf.variable_scope('Generator', reuse=True) as scope:
                Y_tilde_logit, Y_tilde = self.Generator()

            # Discriminator
            with tf.variable_scope('Discriminator', reuse=True) as scope:
                D_logit = self.Discriminator(Y_tilde)

            D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_treat, logits=D_logit))
            # Generator
            G_loss_GAN = -D_loss
            Y_logits = self.is_treat * tf.reshape(Y_tilde_logit[:, 1], [-1, 1]) + (1. - self.is_treat) * tf.reshape(
                Y_tilde_logit[:, 0], [-1, 1])
            G_loss_Factual = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=(Y_logits)))
            G_loss = G_loss_Factual + 1. * G_loss_GAN
        return G_loss

    def IFit(self):
        with tf.variable_scope('model', reuse=True) as scope:
            # Generator
            with tf.variable_scope('Generator', reuse=True) as scope:
                Y_tilde_logit, Y_tilde = self.Generator()

            # Inference
            with tf.variable_scope('Inference', reuse=True) as scope:
                Y_hat_logit = self.Inference()
                Y_hat = tf.nn.sigmoid(Y_hat_logit)

            # Inference
            I_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.is_treat * self.label + (1. - self.is_treat) * tf.reshape(Y_tilde[:, 1], [-1, 1]),
                logits=tf.reshape(Y_hat_logit[:, 1], [-1, 1])
            ))
            I_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=(1. - self.is_treat) * self.label + self.is_treat * tf.reshape(Y_tilde[:, 0], [-1, 1]),
                logits=tf.reshape(Y_hat_logit[:, 0], [-1, 1])
            ))
            I_loss = I_loss1 + I_loss2
            uplift_score = tf.cast(Y_hat[:, 1] - Y_hat[:, 0], tf.float32)
        if self.is_training:
            return I_loss
        else:
            xdl.trace_tf_tensor('uplift_score', uplift_score, scope='train_i')
            return I_loss, tf.reduce_mean(uplift_score)


class TARNET(object):
    def __init__(self, input, is_treat, label, h_dim,
                 is_self, l2_reg, alpha = 1., targeted_reg = True, act_type="elu", is_training=True):
        self.input = input
        self.is_treat = is_treat
        self.label = label
        self.h_dim = h_dim
        self.is_self = is_self
        self.alpha = alpha
        self.targeted_reg = targeted_reg
        self.l2_reg = l2_reg
        self.act_type = act_type
        self.is_training = is_training

    ## Define Tarnet
    def MakeTarnet(self):
        """ Neural net predictive model. The tarnet has three heads

        Args:
        Returns:
            - D_logit: estimated potential outcomes
        """
        # MLP parts
        D_fc1 = utils.fc(self.input, self.h_dim, act_type=self.act_type, is_training=self.is_training, init_w_type='norm', scope="d_fc1")
        D_last = D_fc1
        h_dim = self.h_dim
        if self.is_self:
            D_fc2 = utils.fc(D_fc1, int(self.h_dim / 2), act_type=self.act_type, is_training=self.is_training, init_w_type='norm', scope="d_fc2")
            D_last = D_fc2
            h_dim = int(self.h_dim / 2)

        # propensity for treat predict
        t_logit = utils.fc(self.input, 1, act_type="", is_training=self.is_training, init_w_type='norm', scope="t_logit")

        # hypothesis
        y0_fc1 = utils.fc(D_last, int(h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y0_fc1")
        y0_fc2 = utils.fc(y0_fc1, int(h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y0_fc2")

        y1_fc1 = utils.fc(D_last, int(h_dim / 2), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y1_fc1")
        y1_fc2 = utils.fc(y1_fc1, int(h_dim / 4), act_type=self.act_type, is_training=self.is_training,
                          l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y1_fc2")

        # estimated outcome if treat = 0
        y0_logit = utils.fc(y0_fc2, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y0_logit")

        # estimated outcome if treat = 1
        y1_logit = utils.fc(y1_fc2, 1, act_type="", is_training=self.is_training,
                            l2_reg=self.l2_reg, use_bn=False, init_w_type='norm', scope="y1_logit")

        # Epsilon layer
        with tf.variable_scope('Epsilon', reuse=tf.AUTO_REUSE) as scope:
            dl = tf.get_variable(name='epsilon', shape=[1, 1],
                                 initializer=tf.random_normal_initializer(),
                                 dtype=tf.float32)
            epsilons = dl * tf.ones_like(t_logit)[:, 0:1]

        D_logit = tf.concat(values=[y0_logit, y1_logit, t_logit, epsilons], axis=1)

        return D_logit

    def TarnetFit(self):
        ## Structure
        # Logits
        D_logit = self.MakeTarnet()

        # Losses
        concat_true = tf.concat(values=[self.label, self.is_treat], axis=1)
        if self.targeted_reg:
            loss = utils.make_tarreg_loss(self.alpha, dragonnet_loss=utils.dragonnet_loss_binarycross)
        else:
            loss = utils.dragonnet_loss_binarycross
        D_loss = loss(concat_true, D_logit)

        return D_loss, D_logit
