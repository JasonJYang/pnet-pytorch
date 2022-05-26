import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import scipy
import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
from os.path import join
from prostate_cancer.data_reader import ProstateDataPaper
from pre import get_processor
from reactome.reactome import ReactomeNetwork

class ProstateDataLoader():
    def __init__(self, data_type, params, batch_size, shuffle, num_workers, logger,
                 eval_dataset=True, pre_params=None):
        self.params = params
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.logger = logger
        self.eval_dataset = eval_dataset
        self.pre_params = pre_params

        if data_type == 'prostate_paper':
            params['logger'] = self.logger
            self.data_reader = ProstateDataPaper(**params)
        else:
            self.logger.error('unsupported data type')
            raise ValueError('unsupported data type')

    def get_reactome(self):
        return ReactomeNetwork(data_dir=join(self.params['data_dir'], 'reactome'))

    def get_train_validate_test(self):
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.data_reader.get_train_validate_test()
        # combine training and validation datasets
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = list(info_train) + list(info_validate)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    def get_features_genes(self):
        x, y, info, columns = self.get_data()
        features = columns
        if hasattr(columns, 'levels'):
            genes = columns.levels[0]
        else:
            genes = columns
        return features, genes

    def get_dataloader(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.get_train_validate_test()
        if self.eval_dataset:
            x_t = x_test
            y_t = y_test
            info_t = info_test
        else:
            x_t = np.concatenate((x_test, x_validate))
            y_t = np.concatenate((y_test, y_validate))
            info_t = info_test.append(info_validate)
        
        self.logger.info('x_train {} y_train {} '.format(x_train.shape, y_train.shape))
        self.logger.info('x_test {} y_test {} '.format(x_t.shape, y_t.shape))

        self.logger.info('preprocessing....')
        x_train, x_test = self.preprocess(x_train, x_t)

        train_dataloader = self.create_dataloader(x=x_train, y=y_train)
        validate_dataloader = self.create_dataloader(x=x_validate, y=y_validate)
        test_dataloader = self.create_dataloader(x=x_t, y=y_t)

        return train_dataloader, validate_dataloader, test_dataloader
    
    def preprocess(self, x_train, x_test):
        self.logger.info('preprocessing....')
        proc = get_processor(self.pre_params)
        if proc:
            proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)
            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test

    def create_dataloader(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        dataset = Data.TensorDataset(x, y)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=self.num_workers)
        return dataloader
