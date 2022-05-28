import logging
import pandas as pd
from data_loaders import ProstateDataLoader

logger = logging.getLogger('data_loaders_test')

selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'
selected_samples = 'samples_with_fusion_data.csv'
data_params = {'data_type': 'prostate_paper',
               "batch_size": 50,
               "shuffle": True,
               "num_workers": 2,
               "eval_dataset": False,
               "pre_params": None,
               "logger": logger,
               'params': {
                   "data_dir": "../data/",
                   "data_type": ["mut_important", "cnv_del", "cnv_amp"],
                   "account_for_data_type": None,
                   "cnv_levels": 3,
                   "cnv_filter_single_event": True,
                   "mut_binary": True,
                   "selected_genes": "tcga_prostate_expressed_genes_and_cancer_genes.csv",
                   "combine_type": "union",
                   "use_coding_genes_only": True,
                   "drop_AR": False,
                   "balanced_data": False,
                   "cnv_split": False,
                   "shuffle": False,
                   "selected_samples": None,
                   "training_split": 0
                }
               }

data_adapter = ProstateDataLoader(**data_params)
x, y, info, columns = data_adapter.get_data()

print(x.shape, y.shape, len(columns), len(info))
# x.shape: (1011, 27687), y.shape: (1011, 1), len(columns)=27687 (genes), len(info)=1011 (samples)

x_train, x_test, y_train, y_test, info_train, info_test, columns = data_adapter.get_train_test()
x_train_df = pd.DataFrame(x_train, columns=columns, index=info_train)

print(columns.levels)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.sum().sum())

x, y, info, columns = data_adapter.get_data()
x_df = pd.DataFrame(x, columns=columns, index=info)
print(x_df.shape)
print(x_df.sum().sum())

features, genes = data_adapter.get_features_genes()
print('features: {}, genes: {}'.format(features, genes))