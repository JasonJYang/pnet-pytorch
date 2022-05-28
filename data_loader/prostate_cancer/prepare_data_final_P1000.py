import pandas as pd
from os.path import join

class P1000():
    def __init__(self, data_dir):
        self.processed_dir = join(data_dir, 'processed')
        self.data_dir = join(data_dir, 'raw_data')
        
        # remove silent and intron mutations
        self.filter_silent_muts = False
        self.filter_missense_muts = False
        self.filter_introns_muts = False
        self.keep_important_only = True
        self.truncating_only = False

        self.ext = ""
        if self.keep_important_only:
            self.ext = 'important_only'
        if self.truncating_only:
            self.ext = 'truncating_only'
        if self.filter_silent_muts:
            self.ext = "_no_silent"
        if self.filter_missense_muts:
            self.ext = self.ext + "_no_missense"
        if self.filter_introns_muts:
            self.ext = self.ext + "_no_introns"

        self.prepare_design_matrix_crosstable()
        self.prepare_cnv()
        self.prepare_response()
        self.prepare_cnv_burden()
        print('Done')

    def prepare_design_matrix_crosstable(self):
        print('preparing mutations ...')

        filename = '41588_2018_78_MOESM4_ESM.txt'
        id_col = 'Tumor_Sample_Barcode'
        df = pd.read_csv(join(self.data_dir, filename), sep='\t', low_memory=False, skiprows=1)
        print('mutation distribution')
        # there are multiple variant classifications
        print(df['Variant_Classification'].value_counts())

        if self.filter_silent_muts:
            df = df[df['Variant_Classification'] != 'Silent']
        if self.filter_missense_muts:
            df = df[df['Variant_Classification'] != 'Missense_Mutation']
        if self.filter_introns_muts:
            df = df[df['Variant_Classification'] != 'Intron']

        # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
        exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
        if self.keep_important_only:
            # remove the variants in exclude
            df = df[~df['Variant_Classification'].isin(exclude)]
        if self.truncating_only:
            include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
            df = df[df['Variant_Classification'].isin(include)]
        # the columns of the rearange table are the genes, the rows are the samples and the values denote that whether one sample has this mutation on this gene
        df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                                  aggfunc='count')
        df_table = df_table.fillna(0)
        total_numb_mutations = df_table.sum().sum()

        number_samples = df_table.shape[0]
        print('number of mutations', total_numb_mutations, total_numb_mutations / (number_samples + 0.0))
        filename = join(self.processed_dir, 'P1000_final_analysis_set_cross_' + self.ext + '.csv')
        df_table.to_csv(filename)

    def prepare_response(self):
        print('preparing response ...')
        filename = '41588_2018_78_MOESM5_ESM.xlsx'
        df = pd.read_excel(join(self.data_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
        response = pd.DataFrame()
        response['id'] = df['Patient.ID']
        response['response'] = df['Sample.Type']
        response['response'] = response['response'].replace('Metastasis', 1)
        response['response'] = response['response'].replace('Primary', 0)
        response = response.drop_duplicates()
        # the columns are 1 (Metastasis) or 0 (Primary) and the rows are the samples 
        response.to_csv(join(self.processed_dir, 'response_paper.csv'), index=False)

    def prepare_cnv(self):
        print('preparing copy number variants ...')
        filename = '41588_2018_78_MOESM10_ESM.txt'
        df = pd.read_csv(join(self.data_dir, filename), sep='\t', low_memory=False, skiprows=1, index_col=0)
        df = df.T
        # the columns are the genes and the rows are the samples and the values denote the copy number of variants
        df = df.fillna(0.)
        filename = join(self.processed_dir, 'P1000_data_CNA_paper.csv')
        df.to_csv(filename)

    def prepare_cnv_burden(self):
        print('preparing copy number burden ...')
        filename = '41588_2018_78_MOESM5_ESM.xlsx'
        df = pd.read_excel(join(self.data_dir, filename), skiprows=2, index_col=1)
        cnv = df['Fraction of genome altered']
        # the columns are the fraction of genome altered and the rows are the samples
        filename = join(self.processed_dir, 'P1000_data_CNA_burden.csv')
        cnv.to_frame().to_csv(filename)

p1000 = P1000(data_dir='../../data/prostate')