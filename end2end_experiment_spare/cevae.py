# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf
import xdl
import xdl_runner
from xdl.python.training.export import get_latest_ckpt_v2
from xdl.python.training.saver import Saver, EmbedCodeConf

from exp_io.data_alipub import AliBrandLiftPub
from helper.models import CEVAE

## naming convertion: P_A_BC -> P(A|B, C)
### Generate model / Decoder / Model Network
# Note that we do not use discrete features as we all learned to represent all inputs


@xdl.tf_wrapper(clip_grad = 5.0)
def pre_fix_x(is_treat, embeddings, label, is_training=True):
    input = tf.concat(embeddings, axis=1)
    with tf.variable_scope('model', reuse=True) as scope:
        net = CEVAE(input, is_treat, label, 512, 32, 256,
                    False, True, act_type="elu", is_training=is_training)
        loss = net.FixXFit()
        return loss

@xdl.tf_wrapper(clip_grad = 5.0)
def init_model_z(is_treat, embeddings, label, is_training=True):
    input = tf.concat(embeddings, axis=1)
    with tf.variable_scope('model', reuse=True) as scope:
        net = CEVAE(input, is_treat, label, 512, 32, 256,
                    False, True, act_type="elu", is_training=is_training)
        loss = net.ZInitFit()
        return loss

@xdl.tf_wrapper(clip_grad = 5.0)
def model(is_treat, embeddings, label, is_training=True):
    print('tensorflow version:' + str(tf.__version__))
    input = tf.concat(embeddings, axis=1)
    with tf.variable_scope('model', reuse=True) as scope:
        net = CEVAE(input, is_treat, label, 512, 32, 256,
                    False, True, act_type="elu", is_training=is_training)
        if is_training:
            loss, mean_z_sample \
            , mean_tx_p, mean_zp_last \
            , mean_X_repr, mean_z_prior \
            , mean_loss_x, mean_loss_y \
            , mean_loss_t, mean_loss_pt \
            , mean_loss_py, mean_loss_z = net.CEVAEFit()
            return loss,mean_z_sample\
                   ,mean_tx_p,mean_zp_last\
                   ,mean_X_repr,mean_z_prior\
                   ,mean_loss_x,mean_loss_y\
                   ,mean_loss_t,mean_loss_pt\
                   ,mean_loss_py,mean_loss_z
        else:
            loss, ate = net.CEVAEPred()
            print("is_treat shape:", is_treat.get_shape().as_list())

            print("ate:", ate.get_shape().as_list())

            return loss, ate


def train_xdl_runner(batch):
    pass

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = AliBrandLiftPub(train_dir_name='data_dir', iter_name='iter', is_training=True)
    lr = 0.001 / data_iter.worker_num
    emb_size = 16

    repx_names = ['model/rep_net/r_fc1/fc/kernel', 'model/share_net/r_fc1/fc/bias',
                  'model/rep_net/x_repr/fc/kernel', 'model/share_net/x_repr/fc/bias',
                  'model/rep_net/x_repr_logit/fc/kernel', 'model/share_net/x_repr_logit/fc/bias',
                 ]

    ## fix x train
    # init/set data io
    print("training x representation stage")
    data_iter.set_odps_train_io(iter_name='init', epochs=2)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    emb_list = data_iter.get_embedding_list(emb_dim=emb_size, batch=batch)

    with xdl.model_scope('train1'):
        loss_x = pre_fix_x(is_treat, emb_list, labels, True)
        train_op_x = xdl.Adam(lr).optimize()
        log_hook_x = xdl.LoggerHook(loss_x, "loss:{0}", 10)
        sess1 = xdl.TrainSession(hooks=[log_hook_x])
    while not sess1.should_stop():
        sess1.run([loss_x, train_op_x])

    ## cevae train
    # init/set data io
    print("training x representation stage")
    data_iter.set_odps_train_io(iter_name='iter2', epochs=2)
    batch2 = data_iter.read_batch()
    labels = data_iter.get_cvs(batch2)
    is_treat = data_iter.get_treat(batch2)
    sparse_list = data_iter.get_sparse_list()
    emb_list2 = data_iter.get_embedding_list(emb_dim=emb_size, batch=batch2)

    with xdl.model_scope('train2'):
        losses, mean_z_sample \
        , mean_tx_p, mean_zp_last \
        , mean_X_repr, mean_z_prior \
        , mean_loss_x, mean_loss_y \
        , mean_loss_t, mean_loss_pt \
        , mean_loss_py, mean_loss_z = model(is_treat, emb_list2, labels, True)
        fea_vars = []
        for var in xdl.trainable_variables():
            if var.name not in sparse_list and var.name not in repx_names:
                fea_vars.append(var)
        train_op2 = xdl.Adam(lr).optimize(var_list=fea_vars)

        log_hook2 = xdl.LoggerHook(losses, "loss:{0}", 10)
        log_hook3 = xdl.LoggerHook(mean_z_sample, "mean_z_sample:{0}", 10)
        log_hook4 = xdl.LoggerHook(mean_tx_p, "mean_tx_p:{0}", 10)
        zp_last = xdl.LoggerHook(mean_zp_last, "mean_zp_last:{0}", 10)
        zp_prior = xdl.LoggerHook(mean_z_prior, "mean_z_prior:{0}", 10)
        X_repr = xdl.LoggerHook(mean_X_repr, "mean_X_repr:{0}", 10)

        log_loss_x = xdl.LoggerHook(mean_loss_x, "mean_loss_x:{0}", 10)
        log_loss_y = xdl.LoggerHook(mean_loss_y, "mean_loss_y:{0}", 10)
        log_loss_t = xdl.LoggerHook(mean_loss_t, "mean_loss_t:{0}", 10)
        log_loss_pt = xdl.LoggerHook(mean_loss_pt, "mean_loss_pt:{0}", 10)
        log_loss_py = xdl.LoggerHook(mean_loss_py, "mean_loss_py:{0}", 10)
        log_loss_z = xdl.LoggerHook(mean_loss_z, "mean_loss_z:{0}", 10)


        # log_hook1 = xdl.LoggerHook(loss_PT, "loss_PT:{0}", 10)

        xdl.trace("loss", losses)
        # summary_hook = get_summary_hook()
        # pre_hook = tf.reduce_mean(logits)
        # prediction_hook = xdl.LoggerHook(pre_hook, "prediction:{0}", 10)
        ckpt_meta = xdl.CheckpointMeta()
        if xdl.get_task_index() == 0:
            ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
            sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook2,log_hook3,log_hook4,zp_last,zp_prior,X_repr,log_loss_x,
                                           log_loss_y,log_loss_t,log_loss_pt ,log_loss_py, log_loss_z])
        else:
            sess = xdl.TrainSession(hooks=[log_hook2,log_hook3,log_hook4,zp_last,zp_prior,X_repr
                                    ,log_loss_x,
                                    log_loss_y, log_loss_t, log_loss_pt, log_loss_py, log_loss_z])
    while not sess.should_stop():
        ret = sess.run([losses, train_op2])


def get_summary_hook():
    summary_config = xdl.get_config("summary")
    if summary_config is None:
        raise ValueError("summary config is None")
    output_dir = summary_config.get("output_dir", None)
    if output_dir is None:
        raise ValueError("summary output directory not set")
    summary_config.update({"output_dir": output_dir})
    return xdl.TFSummaryHookV2(summary_config)


def add_trace_hook():
    # trace_config = xdl.get_config("trace")
    trace_config = xdl.get_config('tracer.bin')

    if trace_config is None:
        return
    output_dir = trace_config.get("output_dir", None)
    if output_dir is None:
        raise ValueError("trace output directory not set")
    app_id = xdl.get_app_id()
    if app_id in output_dir:
        output_dir = output_dir.rstrip("/") + "/" + "worker_" + str(xdl.get_task_index())
    else:
        output_dir = output_dir.rstrip("/") + "/" + app_id + "/" + "worker_" + str(xdl.get_task_index())
    trace_config.update({"output_dir": output_dir})
    is_chief = trace_config.get("is_chief", False)
    if xdl.get_task_index() == 0 or is_chief is not True:
        return xdl.TraceHook(trace_config, scope='train2')


def predict():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()
    ckpt_dir = xdl.get_config("checkpoint", "output_dir")
    out_dir = get_latest_ckpt_v2(ckpt_dir)
    model_version = os.path.basename(out_dir)
    if xdl.get_task_index() == 0:
        saver = xdl.Saver(ckpt_dir)

    data_iter = AliBrandLiftPub(test_dir_name='data_dir', iter_name='iter', is_training=False)
    emb_size = 16

    data_iter.set_odps_test_io(iter_name='init', epochs=1)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    sparse_list = data_iter.get_sparse_list()
    emb_list = data_iter.get_embedding_list(emb_dim=emb_size, batch=batch)

    # print("my feature list:", fea_list)

    #########################################################################
    # labels = label_cvs
    # labels = label_vst

    with xdl.model_scope('train2'):

        losses,ate = model(is_treat, emb_list, labels, False)

        xdl.trace_tensor('sample_id', batch['skbuf'], scope='train2')
        for k, i in enumerate(sparse_list):
            xdl.trace_tensor(i, emb_list[k], scope='train2')
        xdl.trace_tensor("is_treat", is_treat, scope='train2')
        xdl.trace_tensor('label', labels, scope='train2')
        xdl.trace_tensor('is_exposure', is_exp, scope='train2')
        hooks = []
        log1_hook = xdl.LoggerHook(losses, "loss3:{0}", 10)
        log2_hook = xdl.LoggerHook(ate, "ate3:{0}", 10)
        trace_hook = add_trace_hook()
        if trace_hook is not None:
            hooks.append(trace_hook)
        hooks.append(log1_hook)
        hooks.append(log2_hook)
        sess = xdl.TrainSession(hooks=hooks)
        while not sess.should_stop():
            ret = sess.run([losses, ate])


def model_export():
    print('export finish')


job_type = xdl.get_config("job_type")
job_type_io = xdl.get_config("job_type_io")
if 'train_xdl_runner' in job_type_io:
    xdl_runner.worker_do(train_xdl_runner)
elif 'train_xdl_io' in job_type_io:
    train_xdl_io()
elif 'auc_xdl_runner' in job_type_io:
    xdl_runner.worker_do(predict)
elif 'auc_xdl_io' in job_type_io:
    predict()
elif 'model_export' in job_type_io:
    model_export()
else:
    raise RuntimeError('Invalid %s job type' % job_type)