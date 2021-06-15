# -*- coding: utf-8 -*-
import xdl
import xdl_runner
import tensorflow as tf
import numpy as np
import os
from xdl.python.training.export import get_latest_ckpt_v2

from exp_io.data_criteo import CriteoUplift1
from helper.models import GANITE

@xdl.tf_wrapper()
def model_d(is_treat, embeddings, label, hidden_dim = 64, is_training=True):
    input = tf.concat(embeddings, axis=1)
    net = GANITE(input, is_treat, label, hidden_dim,
                 False, act_type="relu", is_training=is_training, trainable=True)
    return net.DFit()


@xdl.tf_wrapper()
def model_g(is_treat, embeddings, label,hidden_dim = 64, is_training=True):
    input = tf.concat(embeddings, axis=1)
    net = GANITE(input, is_treat, label, hidden_dim,
                 False, act_type="relu", is_training=is_training, trainable=True)
    return net.GFit()


@xdl.tf_wrapper()
def model_i(is_treat, embeddings, label, hidden_dim = 64,is_training=True):
    input = tf.concat(embeddings, axis=1)
    net = GANITE(input, is_treat, label, hidden_dim,
                 False, act_type="relu", is_training=is_training, trainable=True)
    return net.IFit()

def train_xdl_runner(batch):
    pass

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()

    data_iter = CriteoUplift1(train_dir_name='data_dir', test_dir_name='test_data_dir', iter_name='iter')
    lr = 0.001 / data_iter.worker_num
    # learning_rate = 0.015
    # seq_dim = 12

    generator_params = ['model/Generator/g_fc1/fc/kernel', 'model/Generator/g_fc1/fc/bias',
                        'model/Generator/g_fc2/fc/kernel', 'model/Generator/g_fc2/fc/bias',
                        'model/Generator/g_fc31/fc/kernel', 'model/Generator/g_fc31/fc/bias',
                        'model/Generator/g_fc32/fc/kernel', 'model/Generator/g_fc32/fc/bias',
                        'model/Generator/g_logit1/fc/kernel', 'model/Generator/g_logit1/fc/bias',
                        'model/Generator/g_fc41/fc/kernel', 'model/Generator/g_fc41/fc/bias',
                        'model/Generator/g_fc42/fc/kernel', 'model/Generator/g_fc42/fc/bias',
                        'model/Generator/g_logit2/fc/kernel', 'model/Generator/g_logit2/fc/bias',
                        ]

    discriminator_params = ['model/Discriminator/d_fc1/fc/kernel', 'model/Discriminator/d_fc1/fc/bias',
                            'model/Discriminator/d_fc2/fc/kernel', 'model/Discriminator/d_fc2/fc/bias',
                            'model/Discriminator/d_fc3/fc/kernel', 'model/Discriminator/d_fc3/fc/bias',
                            'model/Discriminator/d_fc4/fc/kernel', 'model/Discriminator/d_fc4/fc/bias',
                            'model/Discriminator/d_logit/fc/kernel', 'model/Discriminator/d_logit/fc/bias',
                            ]

    inference_params = ['model/Inference/i_fc1/fc/kernel', 'model/Inference/i_fc1/fc/bias',
                        'model/Inference/i_fc2/fc/kernel', 'model/Inference/i_fc2/fc/bias',
                        'model/Inference/i_fc31/fc/kernel', 'model/Inference/i_fc31/fc/bias',
                        'model/Inference/i_fc32/fc/kernel', 'model/Inference/i_fc32/fc/bias',
                        'model/Inference/i_logit1/fc/kernel', 'model/Inference/i_logit1/fc/bias',
                        'model/Inference/i_fc41/fc/kernel', 'model/Inference/i_fc41/fc/bias',
                        'model/Inference/i_fc42/fc/kernel', 'model/Inference/i_fc42/fc/bias',
                        'model/Inference/i_logit2/fc/kernel', 'model/Inference/i_logit2/fc/bias',
                        ]

    fea_names = []

    fea_vars = []
    model_d_vars = []
    model_g_vars = []
    model_i_vars = []


    # Start training Generator and Discriminator
    print('Start training Generator and Discriminator')
    # train Discriminator
    # init/set data io

    data_iter.set_odps_train_io(iter_name='reader', epochs=6)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    fea_list = data_iter.get_dens_feas(batch)

    #####################################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train_d'):
        d_loss = model_d(is_treat, fea_list, labels, is_training=True)
        for var in xdl.trainable_variables():
            if var.name in fea_names:
                fea_vars.append(var)
            elif var.name in discriminator_params:
                model_d_vars.append(var)
        model_d_vars += fea_vars
        train_op_d = xdl.Adam(lr).optimize(model_d_vars)
        # auc_hook = xdl.LoggerHook(mse, "mse:{0}", 10)
        log_hook = xdl.LoggerHook(d_loss, "loss:{0}", 10)
        sess_d = xdl.TrainSession(hooks=[log_hook])
    with xdl.model_scope('train_g'):
        g_loss = model_g(is_treat, fea_list, labels, is_training=True)
        for var in xdl.trainable_variables():
            if var.name in fea_names:
                fea_vars.append(var)
            elif var.name in generator_params:
                model_g_vars.append(var)
        model_g_vars += fea_vars
        train_op_g = xdl.Adam(lr).optimize(model_g_vars)
        log_hook = xdl.LoggerHook(g_loss, "loss:{0}", 10)
        sess_g = xdl.TrainSession(hooks=[ log_hook])
    # while (not sess_d.should_stop() and not sess_g.should_stop() ):
    # count = 10000
    while (not sess_d.should_stop() and not sess_g.should_stop()):
        # count -= 1
        for _ in range(2):
            sess_d.run([d_loss, train_op_d])
        sess_g.run([g_loss, train_op_g])

    print('Start training Inference')
    data_iter.set_odps_train_io(iter_name = 'iter2', epochs = 3)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    fea_list = data_iter.get_dens_feas(batch)

    #####################################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train_i'):
        i_loss = model_i(is_treat, fea_list, labels, is_training=True)
        for var in xdl.trainable_variables():
            if var.name in fea_names:
                fea_vars.append(var)
            elif var.name in inference_params:
                model_i_vars.append(var)
        model_i_vars += fea_vars
        train_op_i = xdl.Adam(lr).optimize(model_i_vars)
        log_hook_i = xdl.LoggerHook(i_loss, "loss:{0}", 10)
        # sess_i = xdl.TrainSession(hooks=[log_hook])

        ckpt_meta = xdl.CheckpointMeta()
        if xdl.get_task_index() == 0:
            ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
            sess_i = xdl.TrainSession(hooks=[ckpt_hook, log_hook_i,ckpt_hook])
        else:
            sess_i = xdl.TrainSession(hooks=[log_hook_i])
    while not sess_i.should_stop():
        sess_i.run([i_loss, train_op_i])


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
        return xdl.TraceHook(trace_config, scope='train_i')


def predict():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()
    ckpt_dir = xdl.get_config("checkpoint", "output_dir")
    out_dir = get_latest_ckpt_v2(ckpt_dir)
    model_version = os.path.basename(out_dir)
    if xdl.get_task_index() == 0:
        saver = xdl.Saver(ckpt_dir)

    data_iter = CriteoUplift1(test_dir_name='data_dir', iter_name='iter')
    print('Start predict Inference')
    data_iter.set_odps_test_io(iter_name='reader3', epochs=1)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)
    dense_fea_list = data_iter.get_dens_fea_list()

    ################################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train_i'):
        ii_loss, uplift_score = model_i(is_treat, fea_list, labels, is_training=False)
        xdl.trace_tensor('sample_id', batch['skbuf'],scope='train_i')
        for k, i in enumerate(dense_fea_list):
            xdl.trace_tensor(i, fea_list[k], scope='train_i')
        xdl.trace_tensor('label_cvs', label_cvs, scope='train_i')
        xdl.trace_tensor('label_vst', label_vst, scope='train_i')
        xdl.trace_tensor('is_treat', is_treat, scope='train_i')
        xdl.trace_tensor('is_exposure', is_exp, scope='train_i')
        hooks = []
        log1_hook = xdl.LoggerHook(ii_loss, "loss:{0}", 10)
        log2_hook = xdl.LoggerHook(uplift_score, "uplift_score:{0}", 10)
        trace_hook = add_trace_hook()
        if trace_hook is not None:
            hooks.append(trace_hook)
        hooks.append(log1_hook)
        hooks.append(log2_hook)
        sess_i2 = xdl.TrainSession(hooks=hooks)
        while not sess_i2.should_stop():
            ret = sess_i2.run([ii_loss, uplift_score])


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
    pass
else:
    raise RuntimeError('Invalid %s job type' % job_type)
