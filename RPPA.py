import pandas as pd
import numpy as np
import sys
import os
import mygene
from sklearn.preprocessing import LabelEncoder
import random
from scipy import stats

rppa = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/GDC_data/TCGA-RPPA-pancan-clean.txt",sep="\t", index_col=0)
rppa = rppa.drop(['TumorType'], axis=1)
rppa.to_csv(r"/home/SENSETIME/chenfeiyang/data1/data/RPPA.csv")
