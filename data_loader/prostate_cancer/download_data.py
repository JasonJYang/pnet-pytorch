import os
import urllib.request
from os.path import join, basename, dirname

prostate_dir = '../../data/prostate'
processed_dir = 'processed'
data_dir = 'raw_data'

processed_dir = join(prostate_dir, processed_dir)
data_dir = join(prostate_dir, data_dir)


def download_data():
    print ('downloading data files')
    # P1000 data
    file2 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM6_ESM.xlsx'
    file1 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM4_ESM.txt'
    file3 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'
    file4 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'
    file5 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM5_ESM.xlsx'

    # Met500 files 'https://www.nature.com/articles/nature23306'

    links = [file1, file2, file3, file4, file5]

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for link in links:
        print ('downloading file {}'.format(link))
        filename = join(data_dir, basename(link))
        with open(filename, 'wb') as f:
            f.write(urllib.request.urlopen(link).read())
            f.close()

download_data()
