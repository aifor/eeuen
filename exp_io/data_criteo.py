# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import xdl
import xdl_runner
from xdl.python.training.export import get_latest_ckpt_v2


class CriteoUplift1(object):
    def __init__(self, train_dir_name = None, test_dir_name = None, iter_name='iter'):
        self.iter_name = iter_name
        train_dir = xdl.get_config('reader', train_dir_name)
        test_dir = xdl.get_config('reader', test_dir_name)
        if not isinstance(train_dir, list) and train_dir_name is not None:
            train_dir = sorted(train_dir.split(','))
        if not isinstance(test_dir, list) and test_dir_name is not None:
            test_dir = sorted(test_dir.split(','))

        self.train_dir = train_dir
        self.test_dir = test_dir

        ## columns including features and label/treat
        self.sparse_feas = []
        self.dense_feas = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
        self.labels = ['conversion', 'visit', 'exposure']
        self.treats = ['treatment']

        ## targets
        self.cvs_key = 'conversion'
        self.vst_key = 'visit'
        self.exp_key = 'exposure'

        self.treat_key = 'treatment'

        ## parameter
        self.num_epochs = int(xdl.get_config("reader", "num_epochs"))
        self.num_threads = int(xdl.get_config("reader", "io_thread"))
        self.batch_size = int(xdl.get_config("reader", "batch_size"))
        self.label_count = int(xdl.get_config("reader", "label_count"))
        self.worker_num = xdl.get_task_num()

        ## data io
        self.data_io = None

    def _sparse_read(self, reader, name):
        reader.feature(name=name, type=xdl.features.sparse, table=0, serialized=True)

    def _dense_read(self, reader, name, nvec):
        reader.feature(name=name, type=xdl.features.dense, nvec=nvec)


    def _set_odps_data_io(self, iter_name=None, epochs=1, data_dir = None):
        if iter_name is None:
            iter_name = self.iter_name
        self.data_io = xdl.DataIO(iter_name, file_type=xdl.parsers.column, fs_type=xdl.fs.odps_table, namenode="",
                             enable_state=True)
        sharding = xdl.OdpsTableSharding(self.data_io.fs())
        if data_dir is None:
            data_dir = self.train_dir

        for path in data_dir:
            #   data = path
            sharding.add_path(path)
            print('data path,' + path)

        # sharding.add_path(xdl.get_config('reader', 'data_dir'))
        paths = sharding.partition(rank=xdl.get_task_index(), size=xdl.get_task_num())
        self.data_io.add_path(paths)

        self.data_io.allow_missing(True)

        # 定义sparse特征
        # sparse_configs = sparse_list#get_fg_config()
        # sparse_feature(data_io, sparse_configs)
        for sparse in self.sparse_feas:
            self._sparse_read(self.data_io, sparse)
        # 新增dense_feature
        for dense_fea in self.dense_feas:
            self._dense_read(self.data_io, dense_fea, nvec=1)
        for label in self.labels:
            self._dense_read(self.data_io, label, nvec=1)
        for treat in self.treats:
            self._dense_read(self.data_io, treat, nvec=1)
        # dense_read(data_io, treat_fea[0], 1)

        # NOTE: 流程自己控制
        self.data_io.epochs(epochs)
        # 读数据线程数
        self.data_io.threads(self.num_threads)
        self.data_io.batch_size(self.batch_size)
        self.data_io.label_count(self.label_count)
        # 是否读入sample_id，在做预测和trace时有用
        self.data_io.keep_skey(True)
        self.data_io.startup()

    def set_odps_train_io(self, iter_name=None, epochs=1):
        self._set_odps_data_io(iter_name, epochs, self.train_dir)

    def set_odps_test_io(self, iter_name=None, epochs=1):
        self._set_odps_data_io(iter_name, epochs, self.test_dir)

    def set_train_path(self, train_dir):
        self.train_dir = train_dir

    def set_test_path(self, test_dir):
        self.test_dir = test_dir

    def read_batch(self):
        return self.data_io.read()

    def get_treat(self, batch_iter):
        return batch_iter[self.treat_key]

    def get_cvs(self, batch_iter):
        return batch_iter[self.cvs_key]

    def get_vst(self, batch_iter):
        return batch_iter[self.vst_key]

    def get_exp(self, batch_iter):
        return batch_iter[self.exp_key]

    def get_dens_feas(self, batch_iter):
        dense_feas = []
        for dense in self.dense_feas:
            dense_feas.append(batch_iter[dense])
        return dense_feas
    def get_dens_fea_list(self):
        return self.dense_feas

    def embedding(self, key, sparse_name, emb_dim, batch_iter, is_training):
        if is_training:
            feature_add_probability = 1.0
        else:
            feature_add_probability = 0.0
        hashsize = 1024
        emb = xdl.embedding(key,
                            batch_iter[sparse_name],
                            xdl.TruncatedNormal(stddev=0.001),
                            emb_dim,
                            hashsize,
                            'sum',
                            vtype='hash64',
                            feature_add_probability=feature_add_probability)
        return emb
