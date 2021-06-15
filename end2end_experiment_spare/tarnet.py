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
from helper.models import TARNET

@xdl.tf_wrapper()
def model(is_treat, embeddings, label, is_training=True):
    targeted_reg = True
    alpha = 1.0
    input = tf.concat(embeddings, axis=1)
    net = TARNET(input, is_treat, label, 512,
                 True, 0.01, alpha, targeted_reg, act_type="elu", is_training=is_training)
    D_loss, D_logit = net.TarnetFit()

    if is_training:
        return D_loss
    else:
        uplift_score = D_logit[:, 1] - D_logit[:, 0]
        xdl.trace_tf_tensor('uplift_score', uplift_score, scope='train2')
        print("is_treat shape:", is_treat.get_shape().as_list())
        ate = tf.reduce_mean(uplift_score)
        print("ate:", ate.get_shape().as_list())
        return D_loss,ate


def train_xdl_runner(batch):
    pass

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = AliBrandLiftPub(train_dir_name='data_dir', iter_name='iter',is_training=True)
    lr = 0.001 / data_iter.worker_num
    emb_size = 16
    ## train First stage
    # init/set data io
    print("training first stage")
    data_iter.set_odps_train_io(epochs=1)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    spare_list = data_iter.get_sparse_list()
    emb_list = data_iter.get_embedding_list(emb_size, batch, trainable=True)

    # print("my embedding list:", emb_list)
    with xdl.model_scope('train1'):
        losses1 = model(is_treat, emb_list, labels, True)
        train_op1 = xdl.Adam(lr).optimize()
        log_hook = xdl.LoggerHook(losses1, "loss1:{0}", 10)
        sess = xdl.TrainSession(hooks=[log_hook])
    while not sess.should_stop():
        sess.run([losses1, train_op1])

    # train Second stage
    print("training second stage")
    data_iter.set_odps_train_io(iter_name='iter2', epochs=1)
    batch2 = data_iter.read_batch()
    labels = data_iter.get_cvs(batch2)
    is_treat = data_iter.get_treat(batch2)
    spare_list = data_iter.get_sparse_list()
    emb_list2 = data_iter.get_embedding_list(emb_size, batch2, trainable=True)

    # print("my embedding list:", emb_list)
    with xdl.model_scope('train2'):
        losses = model(is_treat, emb_list2, labels, True)
        lr = 1e-5 / data_iter.worker_num
        train_op2 = xdl.Momentum(lr, 0.9, use_nesterov = True).optimize()
        log_hook2 = xdl.LoggerHook(losses, "loss2:{0}", 10)
        xdl.trace("loss", losses)
    ckpt_meta = xdl.CheckpointMeta()
    if xdl.get_task_index() == 0:
        ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
        sess2 = xdl.TrainSession(hooks=[ckpt_hook, log_hook2])
    else:
        sess2 = xdl.TrainSession(hooks=[log_hook2])
    while not sess2.should_stop():
        sess2.run([losses, train_op2])


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

    emb_size = 16

    data_iter = AliBrandLiftPub(test_dir_name='data_dir', iter_name='iter', is_training=False)
    data_iter.set_odps_test_io(epochs=1)
    data_iter.set_training(False)
    batch = data_iter.read_batch()
    labels = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    sparse_list = data_iter.get_sparse_list()
    emb_list = data_iter.get_embedding_list(emb_size, batch, trainable=False)

    print("my feature list:", sparse_list)

    with xdl.model_scope('train2'):

        losses,ate = model(is_treat, emb_list, labels, False)

        xdl.trace_tensor('sample_id', batch['skbuf'], scope='train2')

        for k, i in enumerate(sparse_list):
            xdl.trace_tensor(i, emb_list[k], scope='train2')

        xdl.trace_tensor('labels', labels, scope='train2')
        xdl.trace_tensor('treat', is_treat, scope='train2')
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
 pass

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
