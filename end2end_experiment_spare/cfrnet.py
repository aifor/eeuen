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
from exp_io.data_alipub import AliBrandLiftPub
from helper.models import CFRNET

total_treat_prob = 0.75

@xdl.tf_wrapper()
def model_bnn(is_treat, embeddings, label, total_treat_prob, rep_norm=False, is_training=True):
    with tf.variable_scope('model') as scope:
        input = tf.concat(embeddings, 1)
        net = CFRNET(input, is_treat, label, 512, rep_norm, True, 0.01, act_type="elu", is_training=is_training)
        # total loss
        if is_training:
            tot_loss, B_logit = net.BNNFit(total_treat_prob)
            return tot_loss
        else:
            B0_logit, _ = net.MakeBNN(tf.zeros_like(is_treat))
            B1_logit, _ = net.MakeBNN(tf.ones_like(is_treat))
            uplift_score = B1_logit - B0_logit
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
        net = CFRNET(input, is_treat, label, 512, rep_norm, True, 0.01, act_type="elu", is_training=is_training)
        tot_loss, B_logit,pred_loss,imb_loss = net.CFRFit(total_treat_prob)

        if is_training:
            return tot_loss,pred_loss,imb_loss
        else:
            uplift_score = B_logit[:, 1] - B_logit[:, 0]
            xdl.trace_tf_tensor('is_treat', is_treat)
            xdl.trace_tf_tensor('uplift_score', uplift_score)
            xdl.trace_tf_tensor('label', label[:, :1])
            print("is_treat shape:", is_treat.get_shape().as_list())
            ate = tf.reduce_mean(uplift_score)
            print("ate:", ate.get_shape().as_list())
            return tot_loss, ate


def train_xdl_runner(batch):
    pass

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = AliBrandLiftPub(train_dir_name='data_dir', iter_name='iter', is_training=True)
    lr = 0.001 / data_iter.worker_num
    emb_size = 16

    print("training first stage")
    data_iter.set_odps_train_io(iter_name='iter1', epochs=2)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    emb_list = data_iter.get_embedding_list(emb_size, batch)
    spare_list = data_iter.get_sparse_list()

    total_loss,pred_loss,imb_loss = model(is_treat, emb_list, labels, total_treat_prob, False, True)
    # train_op1 = xdl.RMSProp(lr, decay=0.5).optimize()
    train_op1 = xdl.Adam(lr).optimize()
    log_hook = xdl.LoggerHook(total_loss, "total_loss:{0}", 10)
    log_hook1 = xdl.LoggerHook(pred_loss, "pred_loss:{0}", 10)
    log_hook2 = xdl.LoggerHook(imb_loss, "imb_loss:{0}", 10)

    xdl.trace("total_loss", total_loss)
    ckpt_meta = xdl.CheckpointMeta()
    if xdl.get_task_index() == 0:
        ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
        sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook,log_hook1,log_hook2])
    else:
        sess = xdl.TrainSession(hooks=[log_hook,log_hook1,log_hook2])
    while not sess.should_stop():
        sess.run([total_loss, train_op1])


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
        return xdl.TraceHook(trace_config, scope='train')


def predict():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()
    ckpt_dir = xdl.get_config("checkpoint", "output_dir")
    out_dir = get_latest_ckpt_v2(ckpt_dir)
    model_version = os.path.basename(out_dir)
    if xdl.get_task_index() == 0:
        saver = xdl.Saver(ckpt_dir)

    emb_size = 16

    data_iter = AliBrandLiftPub(test_dir_name='data_dir', iter_name='iter', is_training=False)
    data_iter.set_odps_test_io(epochs=1)
    data_iter.set_training(False)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    emb_list = data_iter.get_embedding_list(emb_size, batch, False)
    sparse_list = data_iter.get_sparse_list()

    print("my feature list:", emb_list)

    with xdl.model_scope('train'):

        losses,ate = model(is_treat, emb_list, labels, total_treat_prob, False, False)

        xdl.trace_tensor('sample_id', batch['skbuf'])
        for k, i in enumerate(sparse_list):
            xdl.trace_tensor(i, emb_list[k])
        xdl.trace_tensor("is_treat_i", is_treat)
        xdl.trace_tensor("labels_i", labels)
        xdl.trace_tensor('is_exposure', is_exp)
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