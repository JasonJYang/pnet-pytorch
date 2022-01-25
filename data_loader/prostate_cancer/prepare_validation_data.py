import os
import pandas as pd
from os.path import join, dirname, exists

class validation():
    def __init__(self, data_dir):
        self.processed_dir = join(data_dir, 'processed')
        self.data_dir = join(data_dir, 'external_validation')
        self.gene_path = join(dirname(data_dir), 'genes')
        self.inputs_dir = join(self.data_dir, 'Met500')

        self.prepare_Met500_mut()
        self.prepare_Met500_cnv()
        # self.prepare_PRAD()
        print('Done')

    def prepare_Met500_mut(self):
        saving_dir = join(self.data_dir, 'Met500')
        if not exists(saving_dir):
            os.makedirs(saving_dir)

        protein_genes = self.get_protein_encoding_genes()
        print('protein_genes', len(protein_genes))
        mut = self.get_design_matrix_mutation(saving_dir)
        genes = set(mut.columns.values)
        common_genes = protein_genes.intersection(genes)
        print('number of genes {}, number of common genes {} '.format(len(genes), len(common_genes)))

        # saving mutation matrix
        mut.to_csv(join(saving_dir, 'Met500_mut_matrix.csv'))


    def prepare_Met500_cnv(self):
        # processing CNV data
        saving_dir = join(self.data_dir, 'Met500')
        protein_genes = self.get_protein_encoding_genes()
        # cnv_genes= pd.read_csv(join(data_dir,'met500_cnv_unique_genes.txt'), header=None)
        cnv_genes = pd.read_csv(join(saving_dir, 'Met500_cnv.txt'), header=0, index_col=0, sep='\t')
        # cnv_genes.columns = ['genes']
        print(cnv_genes.head())
        genes = set(cnv_genes.index)
        common_genes = protein_genes.intersection(genes)
        print('number of cnv genes {} number of encoding genes {} number of comon genes {} '.format(len(genes),
                                                                                                    len(protein_genes),
                                                                                                    len(common_genes)))

    def get_design_matrix_mutation(self, saving_dir):
        filenmae = 'somatic_v4.csv'
        df = pd.read_csv(join(saving_dir, filenmae), sep=',')
        design_matrix = pd.pivot_table(data=df, values='Effect', index='Pipeline_ID', columns='Gene',
                                    aggfunc='count', fill_value=None)
        return design_matrix


    def get_protein_encoding_genes(self):
        df_protein = pd.read_csv(join(self.gene_path, 'HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'), sep='\t',
                                index_col=0)  # 16190
        df_protein.columns = ['start', 'end', 'symbol']
        df = df_protein
        df_other = pd.read_csv(join(self.gene_path, 'HUGO_genes/other.txt'), sep='\t')  # 112
        genes = set(list(df_other['symbol']) + list(df['symbol']))
        return genes


    def prepare_PRAD(self):
        saving_dir = join(self.data_dir, 'PRAD')
        if not exists(saving_dir):
            os.makedirs(saving_dir)

        import zipfile
        path_to_zip_file = join(saving_dir, '41586_2017_BFnature20788_MOESM324_ESM.zip')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            extract_dir = join(saving_dir, 'nature20788-s2')
            zip_ref.extractall(extract_dir)

        path_to_zip_file = join(saving_dir, '41586_2017_BFnature20788_MOESM325_ESM.zip')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(saving_dir)

        extract_dir = join(saving_dir, 'nature20788-s3')
        mut_filename = join(extract_dir, 'SI Data 1 filtered_variants_by_patient.tsv')

        mut_df = pd.read_csv(join(saving_dir, mut_filename), sep='\t')

        CPCG_cols = [c for c in mut_df.columns.values if c.startswith('CPCG')]
        print(len(CPCG_cols))

        exclude = ['upstream', 'downstream', 'intergenic']  # 'intronic', 'ncRNA_intronic'
        cpcg_mutations = mut_df.loc[~mut_df.Location.isin(exclude), ['Gene'] + CPCG_cols]
        xx = cpcg_mutations.groupby('Gene').sum()
        xx.T.to_csv(join(saving_dir, 'mut_matrix.csv'))

        # cnv
        extract_dir = join(saving_dir, 'nature20788-s2')
        cnv_filename = join(extract_dir, 'Supplementary Table 02 - Per-gene CNA analyses.xlsx')
        cna_df = pd.read_excel(join(saving_dir, cnv_filename))
        cna_df_matrix = cna_df.loc[:, ['Symbol'] + CPCG_cols]
        yy = cna_df_matrix.groupby('Symbol').max()
        yy.T.to_csv(join(saving_dir, 'cnv_matrix.csv'))
