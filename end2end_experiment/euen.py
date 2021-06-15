# -*- coding: utf-8 -*-
import xdl
import xdl_runner
import tensorflow as tf
import os
from xdl.python.training.export import get_latest_ckpt_v2
from exp_io.data_criteo import CriteoUplift1
from helper.models import EUEN

@xdl.tf_wrapper()
def model(is_treat, embeddings, label, is_training=True):
    input = tf.concat(embeddings, axis=1)

    net = EUEN(input, is_treat, label, 64, 64, False, 0.01, act_type = "elu", is_training = is_training)
    return net.EUENFit()

def train_xdl_io():
    if xdl.get_task_name() == "ps" or xdl.get_task_name() == "scheduler":
        xdl.current_env().sess_start()
    data_iter = CriteoUplift1(train_dir_name='data_dir', test_dir_name='test_data_dir', iter_name='iter')
    lr = 0.001 / data_iter.worker_num
    # learning_rate = 0.015
    # emb_dim = 12
    # seq_dim = 12

    print('Start training EUEN.')
    # init/set data io

    data_iter.set_odps_train_io(iter_name='reader', epochs=4)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    fea_list = data_iter.get_dens_feas(batch)

    ###################################################################################################
    # labels = label_cvs
    labels = label_vst

    loss, t_tau, mse, mse_op = model(is_treat, fea_list, labels, is_training=True)
    train_op = xdl.Adam(lr).optimize()
    # train_op = xdl.Adagrad(lr).optimize()
    auc_hook = xdl.LoggerHook(mse, "mse:{0}", 10)
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    
    xdl.trace("loss", loss)
    summary_hook = get_summary_hook()
    ckpt_meta = xdl.CheckpointMeta()
    if xdl.get_task_index() == 0:
        ckpt_hook = xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_checkpoint_interval'), meta=ckpt_meta)
        sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook, auc_hook,summary_hook])
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
    summary_config.update({"output_dir": output_dir})
    return xdl.TFSummaryHookV2(summary_config)


def add_trace_hook():
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

    # init/set data io
    data_iter.set_odps_test_io(epochs=1)
    batch = data_iter.read_batch()
    label_vst = data_iter.get_vst(batch)
    label_cvs = data_iter.get_cvs(batch)
    is_treat = data_iter.get_treat(batch)
    is_exp = data_iter.get_exp(batch)
    fea_list = data_iter.get_dens_feas(batch)
    dense_fea_list = data_iter.get_dens_fea_list()

    #####################################################################################
    # labels = label_cvs
    labels = label_vst

    with xdl.model_scope('train'):
        loss, ate, mse, mse_op = model(is_treat,fea_list, labels, is_training=False)
        xdl.trace_tensor('sample_id', batch['skbuf'])

        for k, i in enumerate(dense_fea_list):
            xdl.trace_tensor(i, fea_list[k])

        xdl.trace_tensor('label_cvs', label_cvs)
        xdl.trace_tensor('label_vst', label_vst)
        xdl.trace_tensor('is_treat', is_treat)
        xdl.trace_tensor('is_exposure', is_exp)
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

def train_xdl_runner(batch):
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

