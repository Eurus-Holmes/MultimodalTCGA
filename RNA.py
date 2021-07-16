import pandas as pd
import numpy as np
import sys
import os
import mygene
from sklearn.preprocessing import LabelEncoder
import random
from scipy import stats

rna_data = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/GDC_data/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv",sep="\t", index_col=0)
rna_data = rna_data.apply(np.log1p)
rna_data = rna_data.dropna(axis=0,how='any')
rna_data = rna_data.T
print(rna_data.shape)
# (11069, 16335)
rna_data.to_csv(r"/home/SENSETIME/chenfeiyang/data1/data/GDC_RNA.csv")
