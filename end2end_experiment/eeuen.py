# -*- coding: utf-8 -*-
import xdl
import xdl_runner
import tensorflow as tf
import numpy as np
import time
import os
from xdl.python.training.saver import Saver, EmbedCodeConf
from xdl.python.training.export import get_latest_ckpt_v2
from exp_io.data_criteo import CriteoUplift1
from helper.models import EEUEN

@xdl.tf_wrapper()
def model_exp_pre(is_treat, is_assign, embeddings, label, is_training=True):
    x = tf.concat(embeddings, 1)
    net = EEUEN(x, is_treat, is_assign, label, 64, 64, 64,
                False, 0.001, act_type="elu", is_training=is_training)
    return net.ExpPreFit('train1')

@xdl.tf_wrapper()
def model(is_treat, is_assign, embeddings, label, is_training=True):
    x = tf.concat(embeddings, 1)
    net = EEUEN(x, is_treat, is_assign, label, 64, 64, 64,
                False, 0.001, act_type="elu", is_training=is_training)
    return net.EEUENFit('train2')

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = CriteoUplift1(train_dir_name='data_dir', iter_name='iter')
    lr = 0.001 / data_iter.worker_num
    # learning_rate = 0.015
    # emb_dim = 12
    # seq_dim = 12

    # Start training Generator and Discriminator
    print('Start training EEUEN.')

    # init/set data io
    data_iter.set_odps_train_io(iter_name='iter1', epochs=1)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_assign = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)

    #######################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train1'):
        assign_loss, e_prob = model_exp_pre(is_treat, is_assign, fea_list, labels, is_training=True)
        train_exp_op = xdl.Adam(lr).optimize()
        log_hook = xdl.LoggerHook(assign_loss, "assign_loss:{0}", 10)
        sess_exp = xdl.TrainSession(hooks=[log_hook])
    while not sess_exp.should_stop():
        sess_exp.run([assign_loss, train_exp_op])

    # step two
    data_iter.set_odps_train_io(iter_name='iter2', epochs=4)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_assign = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)

    ########################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train2'):
        loss, logits, mse, mse_op = model(is_treat, is_assign, fea_list, labels, is_training=True)
        # train_op = xdl.SGD(learning_rate).optimize()
        train_op = xdl.Adam(lr).optimize()
        auc_hook = xdl.LoggerHook(mse, "mse:{0}", 10)
        log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    # pre_hook = tf.reduce_mean(logits)
    # prediction_hook = xdl.LoggerHook(pre_hook, "prediction:{0}", 10)
    ckpt_meta = xdl.CheckpointMeta()
    summary_hook = get_summary_hook()
    if xdl.get_task_index() == 0:
        ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
        sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook, auc_hook, summary_hook])
    else:
        sess = xdl.TrainSession(hooks=[log_hook, auc_hook])
    while not sess.should_stop():
        sess.run([loss, train_op])


def get_summary_hook():
    summary_config = xdl.get_config("summary")
    if summary_config is None:
        raise ValueError("summary config is None")
    output_dir = summary_config.get("output_dir", None)
    if output_dir is None:
        raise ValueError("summary output directory not set")
    app_id = xdl.get_app_id()
    if app_id in output_dir:
        output_dir = output_dir.rstrip("/") + "/" + "worker_" + str(xdl.get_task_index())
    else:
        output_dir = output_dir.rstrip("/") + "/" + app_id + "/" + "worker_" + str(xdl.get_task_index())
    summary_config.update({"output_dir": output_dir})
    return xdl.TFSummaryHook(summary_config)


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

    data_iter = CriteoUplift1(test_dir_name='data_dir', iter_name='iter')
    # init/set data io
    data_iter.set_odps_test_io(epochs=1)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)
    dense_fea_list = data_iter.get_dens_fea_list()

    #################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train2'):
        # loss1,loss2,ate,auc,auc_op,auc2, auc_op2 = model(is_treat, emb_list, labels, is_training=False)
        loss,ate,mse,mse_op = model(is_treat, is_exp, fea_list, labels, is_training=False)
        xdl.trace_tensor('sample_id', batch['skbuf'],scope='train2')
        for k, i in enumerate(dense_fea_list):
            xdl.trace_tensor(i, fea_list[k],scope='train2')
        xdl.trace_tensor('label_cvs', label_cvs, scope='train2')
        xdl.trace_tensor('label_vst', label_vst, scope='train2')
        xdl.trace_tensor('is_treat', is_treat, scope='train2')
        xdl.trace_tensor('is_exposure', is_exp, scope='train2')

        hooks = []
        trace_hook = add_trace_hook()
        if trace_hook is not None:
            hooks.append(trace_hook)

        summary_hook = get_summary_hook()
        hooks.append(summary_hook)

        from xdl.python.training.training_utils import get_global_step
        global_step = get_global_step()
        # model_version = xdl.get_config("checkpoint", "version")
        sess = xdl.TrainSession(hooks=hooks)
        while not sess.should_stop():
            ret = sess.run([loss,ate,mse,mse_op])


def model_export():
    pass

job_type = xdl.get_config("job_type")
job_type_io = xdl.get_config("job_type_io")
if 'train_xdl_runner' in job_type_io:
    pass
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


