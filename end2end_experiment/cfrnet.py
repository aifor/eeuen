# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf
import xdl
import xdl_runner
from xdl.python.training.export import get_latest_ckpt_v2
from xdl.python.training.saver import Saver, EmbedCodeConf

import utils
from exp_io.data_criteo import CriteoUplift1
from helper.models import CFRNET

total_treat_prob = 0.85

@xdl.tf_wrapper()
def model_bnn(is_treat, embeddings, label, total_treat_prob, rep_norm=False, is_training=True):
    with tf.variable_scope('model') as scope:
        input = tf.concat(embeddings, 1)
        net = CFRNET(input, is_treat, label, 64, rep_norm, False, 0.01, act_type="elu", is_training=is_training)
        # total loss
        if is_training:
            tot_loss, B_logit = net.BNNFit(total_treat_prob)
            return tot_loss
        else:
            B0_logit, _ = net.MakeBNN(tf.zeros_like(is_treat))
            B1_logit, _ = net.MakeBNN(tf.ones_like(is_treat))
            uplift_score = B1_logit - B0_logit
            # uplift_score = tf.nn.sigmoid(B1_logit) - tf.nn.sigmoid(B0_logit)
            xdl.trace_tf_tensor('is_treat', is_treat)
            xdl.trace_tf_tensor('uplift_score', uplift_score)
            xdl.trace_tf_tensor('label', label[:, :1])
            print("is_treat shape:", is_treat.get_shape().as_list())
            ate = tf.reduce_mean(uplift_score)
            print("ate:", ate.get_shape().as_list())
            return tf.reduce_mean(tf.square(B1_logit - label)), ate

@xdl.tf_wrapper()
def model(is_treat, embeddings, label, total_treat_prob, rep_norm=False, is_training=True):
    with tf.variable_scope('model') as scope:
        input = tf.concat(embeddings, 1)
        net = CFRNET(input, is_treat, label, 64, rep_norm, False, 0.01, act_type="elu", is_training=is_training)
        tot_loss, B_logit = net.CFRFit(total_treat_prob)

        if is_training:
            return tot_loss
        else:
            uplift_score = B_logit[:, 1] - B_logit[:, 0]
            xdl.trace_tf_tensor('uplift_score', uplift_score)
            print("is_treat shape:", is_treat.get_shape().as_list())
            ate = tf.reduce_mean(uplift_score)
            print("ate:", ate.get_shape().as_list())
            return tot_loss, ate


def train_xdl_runner(batch):
    pass

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = CriteoUplift1(train_dir_name='data_dir')
    # lrate = 0.05 / data_iter.worker_num
    # lr = tf.train.exponential_decay(lrate, global_step,
    #                                 1000, 0.95, staircase=True)
    lr = 0.001/ data_iter.worker_num

    # init/set data io
    data_iter.set_odps_train_io(iter_name='reader', epochs=4)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    fea_list = data_iter.get_dens_feas(batch)

    ###############################################################################
    # labels = label_cvs
    labels = label_vst

    loss = model(is_treat, fea_list, labels, total_treat_prob, False, True)

    # train_op1 = xdl.RMSProp(lr, decay=0.5).optimize()
    train_op1 = xdl.Adam(lr).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)

    xdl.trace("loss", loss)
    # summary_hook = get_summary_hook()
    # pre_hook = tf.reduce_mean(logits)
    # prediction_hook = xdl.LoggerHook(pre_hook, "prediction:{0}", 10)
    ckpt_meta = xdl.CheckpointMeta()
    if xdl.get_task_index() == 0:
        ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
        sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook])
    else:
        sess = xdl.TrainSession(hooks=[log_hook])
    while not sess.should_stop():
        sess.run([loss, train_op1])


def get_summary_hook():
    summary_config = xdl.get_config("summary")
    if summary_config is None:
        raise ValueError("summary config is None")
    output_dir = summary_config.get("output_dir", None)
    if output_dir is None:
        raise ValueError("summary output directory not set")
    # from xdl.python.utils.config import *
    # app_id = xdl.get_app_id()
    # if app_id in output_dir:
    #     output_dir = output_dir.rstrip("/") + "/" + "worker_" + str(xdl.get_task_index())
    # else:
    #     output_dir = output_dir.rstrip("/") + "/" + app_id + "/" + "worker_" + str(xdl.get_task_index())
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
        return xdl.TraceHook(trace_config, scope='train')


def predict():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()
    ckpt_dir = xdl.get_config("checkpoint", "output_dir")
    out_dir = get_latest_ckpt_v2(ckpt_dir)
    model_version = os.path.basename(out_dir)
    if xdl.get_task_index() == 0:
        saver = xdl.Saver(ckpt_dir)

    data_iter = CriteoUplift1(test_dir_name='data_dir', iter_name='iter')
    lr = 0.001 / data_iter.worker_num

    # init/set data io
    data_iter.set_odps_test_io(epochs=1)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)
    dense_fea_list = data_iter.get_dens_fea_list()

    #####################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train'):
        losses,ate = model(is_treat, fea_list, labels, total_treat_prob, False, False)

        xdl.trace_tensor('sample_id', batch['skbuf'])
        for k, i in enumerate(dense_fea_list):
            xdl.trace_tensor(i, fea_list[k])
        xdl.trace_tensor("is_treat", is_treat)
        xdl.trace_tensor("label_cvs", label_cvs)
        xdl.trace_tensor("label_vst", label_vst)
        xdl.trace_tensor("is_exposure", is_exp)
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
