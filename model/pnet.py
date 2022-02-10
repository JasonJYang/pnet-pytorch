import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn

from collections import OrderedDict

class Diagonal(nn.Module):
    def __init__(self, output_dim, use_bias=True, input_shape=None):
        super(Diagonal, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        input_dim = input_shape[1]
        self.n_inputs_per_node = input_dim // self.output_dim

        # create parameter
        self.kernel = nn.Parameter(torch.randn(1, input_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [row, genes*3]
        print('input dimensions {}'.format(x.shape))
        mult = x * self.kernel # [row, genes*3]
        mult = torch.reshape(mult, (-1, self.n_inputs_per_node)) # [row*genes, 3]
        mult = torch.sum(mult, dim=1) # [row*genes]
        output = torch.reshape(mult, (-1, self.output_dim)) # [row, genes]

        if self.use_bias:
            output = output + self.bias

        return output

class SparseTF(nn.Module):
    def __init__(self, output_dim, map=None, use_bias=True, input_shape=None):
        super(SparseTF, self).__init__()
        self.output_dim = output_dim
        input_dim = input_shape[1]
        self.map = map
        self.use_bias = use_bias

        self.kernel = nn.Parameter(torch.randn(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        map = self.map.to(inputs.device)
        tt = self.kernel * map
        output = torch.matmul(inputs, tt)
        if self.use_bias:
            output = output + self.bias
        return output


class Dense(nn.Module):
    def __init__(self, output_dim, input_shape=None, use_bias=True):
        super(Dense, self).__init__()
        input_dim = input_shape[1]
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.kernel = nn.Parameter(torch.randn(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.mm(x, self.kernel)
        if self.use_bias:
            x = x + self.bias
        return x

class PNetLayer(nn.Module):
    def __init__(self, mapp, dropout, sparse, use_bias, attention, batch_normal):
        super(PNetLayer, self).__init__()
        self.attention = attention
        self.batch_normal = batch_normal
        n_genes, n_pathways = mapp.shape
        if sparse:
            mapp = self.df_to_tensor(mapp)
            self.hidden_layer = SparseTF(n_pathways, mapp, use_bias=use_bias, input_shape=(None, n_genes))
        else:
            self.hidden_layer = Dense(n_pathways, input_shape=(None, n_genes))
        self.activation = nn.Tanh()
        
        if attention:
            self.attention_prob_layer = Dense(n_pathways, input_shape=(None, n_genes))
            self.attention_activation = nn.Sigmoid()
        
        # testing
        self.decision_layer = Dense(1, input_shape=(None, n_genes))
        if batch_normal:
            self.decision_batch_normal = nn.BatchNorm1d(num_features=n_pathways)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, outcome):
        outcome = self.hidden_layer(outcome)
        outcome = self.activation(outcome)
        if self.attention:
            outcome = self.attention_prob_layer(outcome)
            outcome = self.attention_activation(outcome)
        
        decision_outcome = self.decision_layer(outcome)
        if self.batch_normal:
            decision_outcome = self.decision_batch_normal(decision_outcome)
        
        outcome = self.dropout_layer(outcome)
    
        return outcome, decision_outcome

    def df_to_tensor(self, df):
        tensor = torch.from_numpy(df.to_numpy())
        tensor = tensor.type(torch.FloatTensor)
        return tensor


class PNet(nn.Module):
    def __init__(self, features, genes, n_hidden_layers, direction, dropout, sparse, add_unk_genes, 
                 batch_normal, reactome_network, use_bias=False, shuffle_genes=False, attention=False, 
                 sparse_first_layer=True, logger=None):
        super(PNet, self).__init__()
        self.feature_names = {}
        n_features = len(features)
        n_genes = len(genes)
        self.reactome_network = reactome_network
        self.attention = attention 
        self.batch_normal = batch_normal
        self.n_hidden_layers = n_hidden_layers
        self.logger = logger

        if sparse:
            if shuffle_genes == 'all':
                ones_ratio = float(n_features) / np.prod([n_genes, n_features])
                self.logger.info('ones_ratio random {}'.format(ones_ratio))
                mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
                self.layer1 = SparseTF(n_genes, mapp, use_bias=use_bias, input_shape=(None, n_features))
            else:
                self.layer1 = Diagonal(n_genes, input_shape=(None, n_features), use_bias=use_bias)

        else:
            if sparse_first_layer:
                self.layer1 = Diagonal(n_genes, input_shape=(None, n_features), use_bias=use_bias)
            else:
                self.layer1 = Dense(n_genes, input_shape=(None, n_features), use_bias=use_bias)
        
        self.layer1_activation = nn.Tanh()
        
        if attention:
            self.attention_prob_layer = Diagonal(n_genes, input_shape=(None, n_features))
            self.attention_activation = nn.Sigmoid()
        
        self.dropout1 = nn.Dropout(dropout[0])

        # testing
        self.decision_layer1 = nn.Linear(in_features=n_genes, out_features=1)
        self.batchnorm_layer1 = nn.BatchNorm1d(num_features=n_genes)
        self.decision_activation1 = nn.Sigmoid()

        module_list = []
        if n_hidden_layers > 0:            
            maps = self.get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
            layer_inds = range(1, len(maps))

            print('original dropout', dropout)
            print('dropout', layer_inds, dropout)
            dropouts = dropout[1:]

            for i, mapp in enumerate(maps[0:-1]):
                dropout = dropouts[1]
                names = mapp.index
                if shuffle_genes in ['all', 'pathways']:
                    mapp = self.shuffle_genes_map(mapp)
                n_genes, n_pathways = mapp.shape
                self.logger.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
                print('layer {}, dropout {}'.format(i, dropout))
                
                pnet_layer = PNetLayer(mapp=mapp, 
                    dropout=dropout, sparse=sparse, use_bias=use_bias, attention=attention, batch_normal=batch_normal)
                module_list.append(pnet_layer)

                self.feature_names['h{}'.format(i)] = names

            i = len(maps)
            self.feature_names['h{}'.format(i-1)] = maps[-1].index
        self.module_list = nn.ModuleList(module_list)
        
    def forward(self, inputs):
        decision_outcomes = []
        # inputs: [samples, features]
        outcome = self.layer1(inputs) # [samples, genes]
        outcome = self.layer1_activation(outcome)

        if self.attention:
            attention_probs = self.attention_prob_layer(outcome)
            outcome = outcome * attention_probs
        outcome = self.dropout1(outcome)

        decision_outcome = self.decision_layer1(outcome)
        if self.batch_normal:
            decision_outcome = self.batchnorm_layer1(decision_outcome)
        decision_outcome = self.decision_activation1(decision_outcome)
        decision_outcomes.append(decision_outcome)

        if self.n_hidden_layers > 0:
            for layer in self.module_list:
                outcome, decision_outcome = layer(outcome)
                decision_outcomes.append(decision_outcome)

        return outcome, decision_outcomes

    def get_layer_maps(self, genes, n_levels, direction, add_unk_genes):
        reactome_layers = self.reactome_network.get_layers(n_levels, direction)
        filtering_index = genes
        maps = []
        for i, layer in enumerate(reactome_layers[::-1]):
            print('layer #', i)
            mapp = self.get_map_from_layer(layer)
            filter_df = pd.DataFrame(index=filtering_index)
            print('filtered_map', filter_df.shape)
            filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
            
            print('filtered_map', filter_df.shape)

            # UNK, add a node for genes without known reactome annotation
            if add_unk_genes:
                print('UNK ')
                filtered_map['UNK'] = 0
                ind = filtered_map.sum(axis=1) == 0
                filtered_map.loc[ind, 'UNK'] = 1

            filtered_map = filtered_map.fillna(0)
            print('filtered_map', filter_df.shape)
            # filtering_index = list(filtered_map.columns)
            filtering_index = filtered_map.columns
            self.logger.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
            maps.append(filtered_map)
        return maps

    def get_map_from_layer(self, layer_dict):
        pathways = list(layer_dict.keys())
        print('pathways', len(pathways))
        genes = list(itertools.chain.from_iterable(layer_dict.values()))
        genes = list(np.unique(genes))
        print('genes', len(genes))

        n_pathways = len(pathways)
        n_genes = len(genes)

        mat = np.zeros((n_pathways, n_genes))
        for p, gs in layer_dict.items():
            g_inds = [genes.index(g) for g in gs]
            p_ind = pathways.index(p)
            mat[p_ind, g_inds] = 1

        df = pd.DataFrame(mat, index=pathways, columns=genes)

        return df.T

    def shuffle_genes_map(self, mapp):
        self.logger.info('shuffling')
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        self.logger.info('ones_ratio {}'.format(ones_ratio))
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
        self.logger.info('random map ones_ratio {}'.format(ones_ratio))
        return mapp

    def df_to_tensor(self, df):
        tensor = torch.from_numpy(df.to_numpy())
        tensor = tensor.type(torch.FloatTensor)
        return tensor