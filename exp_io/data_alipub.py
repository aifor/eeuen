# -*- coding: utf-8 -*-

import xdl

class AliBrandLiftPub(object):
    def __init__(self, train_dir_name=None,
                 test_dir_name = None,
                 iter_name='iter', is_training = False):
        self.iter_name = iter_name

        train_dir = xdl.get_config('reader', train_dir_name)
        test_dir = xdl.get_config('reader', test_dir_name)

        if not isinstance(train_dir, list) and train_dir_name is not None:
            train_dir = sorted(train_dir.split(','))
        if not isinstance(test_dir, list) and test_dir_name is not None:
            test_dir = sorted(test_dir.split(','))

        self.train_dir = train_dir
        self.test_dir = test_dir if test_dir_name is not None else None

        ## columns including features and label/treat
        # sparse part
        self.ub_feas = ['101_21','102_21','103_21', '104_21', '111_21', '112_21','601_90','602_90']
        self.up_feas = ['102', '103', '104', '121', '122']
        self.ad_feas = ['201', '203', '204','206','207','209','210','219']
        self.ad_feas_contorl = [ '206', '219']
        self.context_feas = ['301','302','303','311']
        # dense part
        self.exposure_flag = ['401']
        self.treat_flag = ['403','407']
        self.label_flag = ['label']
        self.pctr_fea = []
        self.dense_feas = []


        ## targets
        self.cvs_key = 'label'
        # self.vst_key = 'label'
        self.exp_key = '401'

        self.treat_key = '403'
        self.new_treat_key = '407'

        ## parameter
        self.num_epochs = int(xdl.get_config("reader", "num_epochs"))
        self.num_threads = int(xdl.get_config("reader", "io_thread"))
        self.batch_size = int(xdl.get_config("reader", "batch_size"))
        self.label_count = int(xdl.get_config("reader", "label_count"))
        self.worker_num = xdl.get_task_num()

        ## data io
        self.data_io = None

        ## training
        self.is_train = is_training

    def _sparse_read(self, reader, name):
        reader.feature(name=name, type=xdl.features.sparse, table=0, serialized=True)

    def _dense_read(self, reader, name, nvec):
        reader.feature(name=name, type=xdl.features.dense, nvec=nvec)

    def get_sparse_list(self):
        return self.ub_feas + self.up_feas + self.ad_feas + self.context_feas
    # def get_base_sparse_list(self):
    #     return self.up_feas + self.ad_feas + self.context_feas

    def get_control_list(self):
        return self.ub_feas + self.up_feas + self.ad_feas_contorl + self.context_feas

    def get_dense_list(self):
        return self.dense_feas + self.pctr_fea

    def get_target_flag_list(self):
        return self.exposure_flag + self.treat_flag

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
        # sparse feature部分
        for sparse in self.get_sparse_list():
            self._sparse_read(self.data_io, sparse)
        # dense feature部分
        for dense_fea in self.get_dense_list():
            self._dense_read(self.data_io, dense_fea, nvec = 1)
        # 标记/目标部分
        for target in self.get_target_flag_list():
            self._dense_read(self.data_io, target, nvec = 1)
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

    def set_training(self, is_training=False):
        self.is_train = is_training

    def read_batch(self):
        return self.data_io.read()

    def get_treat(self, batch_iter):
        return batch_iter[self.treat_key]

    def get_new_treat(self, batch_iter):
        return batch_iter[self.new_treat_key]

    def get_cvs(self, batch_iter):
        return batch_iter[self.cvs_key]

    # def get_vst(self, batch_iter):
    #     return batch_iter[self.vst_key]

    def get_exp(self, batch_iter):
        return batch_iter[self.exp_key]

    def get_dens_feas(self, batch_iter):
        dense_feas = []
        for dense in self.get_dense_list():
            dense_feas.append(batch_iter[dense])
        return dense_feas

    def embedding(self, key, sparse_name, emb_dim, batch_iter, trainable = True):
        if self.is_train:
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
                            trainable=trainable,
                            feature_add_probability=feature_add_probability)
        return emb

    def get_embedding_list(self, emb_dim, batch, trainable = True):
        emb_list = []
        for sparse in self.get_sparse_list():
            tmp_embd = self.embedding(sparse, sparse, emb_dim, batch, trainable)
            emb_list.append(tmp_embd)
        return emb_list
