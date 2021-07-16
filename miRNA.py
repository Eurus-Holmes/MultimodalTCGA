import pandas as pd
import numpy as np
import sys
import os
import mygene
from sklearn.preprocessing import LabelEncoder
import random
from scipy import stats

mirna = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/GDC_data/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv", index_col=0)
mirna = mirna.drop(columns=['Correction'])
mirna = mirna.apply(np.log1p)
mirna = mirna.dropna(axis=0,how='any')
mirna = mirna.T
print(mirna.shape)
# (10824, 743)
mirna.to_csv(r"/home/SENSETIME/chenfeiyang/data1/data/GDC_miRNA.csv")
