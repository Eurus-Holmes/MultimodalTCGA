import pandas as pd
import numpy as np
import sys
import os
import mygene
from sklearn.preprocessing import LabelEncoder
import random
from scipy import stats
import gc



sample_label_df = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/GDC_data/Survival_SupplementalTable_S1_20171025_xena_sp",sep="\t")
# sample_label_df = pd.read_csv("Survival_SupplementalTable_S1_20171025_xena_sp",sep="\t")
sample_label_df = sample_label_df[["sample","cancer type abbreviation"]]

sample_label_df["is_primary"] = sample_label_df["sample"].apply(lambda x:"primary" if (x[-2:]=="01" or x[-2:]=="05") else ( "transfer" if (x[-2:]=="06" or x[-2:]=="07") else np.nan))
sample_label_df = sample_label_df[~sample_label_df["is_primary"].isna()]
sample_label_df.columns = ['sample_id', 'cancer_type', 'is_primary']
sample_label_df.set_index(["sample_id"], inplace=True)
print(sample_label_df.shape)
# print(sample_label_df)
# sample_num 12591 -> 10913


# ## DNA RNA 基因panel选取,  特征拼接
# dna rna 的基因与至本的基因 panel(576) 取交集 <br>
# dna 未记录数据 表示未发生变异<br> rna、甲基化 缺失值使用列的均值填充 <br>
# 最终特征文件 gdc_all_data.csv
#

# In[7]:

# gene576 = pd.read_csv(r"./cup/data/zb_Xs_576.csv")
# gene576 = pd.read_csv(r"../tmp_dir/zb_Xs_576.csv", index_col=0)
# zb_gene_panel = list(gene576.columns)

# In[22]:


#  merge dna
# tmp_dna_df = dna_snv_df
# tmp_dna_df = pd.read_csv(r"../data/DNA_VAF_GDC.csv", index_col=0)  # 2min
#
# tmp_dna_gene_names = list(tmp_dna_df.columns)
#
# final_gene_panel_dna = set(tmp_dna_gene_names)
# # 567
#
# final_dna_df = tmp_dna_df[final_gene_panel_dna]  # 10182 rows × 567 columns

# final_dna_df = pd.read_csv(r"../data/DNA_VAF_GDC.csv", index_col=0)
file_path = "/home/SENSETIME/chenfeiyang/data1/data/DNA_VAF_GDC.csv"
final_dna_df = pd.read_csv(file_path, index_col=0)
print("final_dna_df: ", final_dna_df.shape)


# sample_id  TCGA-02-0003-01A ->  TCGA-02-0003-01
final_dna_df.index = map(lambda s: str(s)[:-1], final_dna_df.index)
final_dna_df.columns = map(lambda s: s + '_snv', final_dna_df.columns)
print(final_dna_df.shape)
final_dna_df = pd.merge(sample_label_df, final_dna_df, left_index=True, right_index=True)

print("final_dna_df: ", final_dna_df.shape)
# 9940 rows × 567 columns

# del tmp_dna_df, final_dna_df  # release memory

# In[24]:


# print(len(zb_gene_panel), len(tmp_dna_gene_names))
# print(len(final_gene_panel_dna))
#
# gene_dna_snv = list(set(tmp_dna_gene_names) & set(gene576.columns))
# print(len(gene_dna_snv))

# In[9]:


# merge rna
# tmp_rna_df = rna_t
# tmp_rna_df = pd.read_csv(r"../data/GDC_RNA.csv", index_col=0)  # 10 min
#
# final_gene_panel_rna = set(zb_gene_panel) & set(tmp_rna_df.columns)
#
# final_rna_df = tmp_rna_df[final_gene_panel_rna]  # 11768 rows x 574 columns

# final_rna_df = pd.read_csv(r"../data/GDC_RNA.csv", index_col=0)  # 10 min
final_rna_df = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/GDC_RNA.csv", index_col=0)
print("final_rna_df: ", final_rna_df.shape)
final_rna_df.index = map(lambda s: str(s)[:15], final_rna_df.index)
final_rna_df.columns = map(lambda s: s + '_rna', final_rna_df.columns)

# fill na
# for col_name in list(final_rna_df.columns[final_rna_df.isnull().sum() > 0]):
#     mean_val = final_rna_df[col_name].mean()
#     final_rna_df[col_name].fillna(mean_val, inplace=True)
final_rna_df.fillna(final_rna_df.mean(), inplace=True)

# print(final_dna_df.index)
# print(final_rna_df.index)
final_dna_rna_df = pd.merge(final_dna_df, final_rna_df, left_index=True, right_index=True)

del final_rna_df, final_dna_df  # release memory
gc.collect()
print("final_dna_rna_df: ", final_dna_rna_df.shape)



final_mirna_df = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/GDC_miRNA.csv", index_col=0)
print("final_mirna_df: ", final_mirna_df.shape)
final_mirna_df.index = map(lambda s: str(s)[:15], final_mirna_df.index)
final_mirna_df.columns = map(lambda s: s + '_mirna', final_mirna_df.columns)

# fill na
# for col_name in list(final_rna_df.columns[final_rna_df.isnull().sum() > 0]):
#     mean_val = final_rna_df[col_name].mean()
#     final_rna_df[col_name].fillna(mean_val, inplace=True)
final_mirna_df.fillna(final_mirna_df.mean(), inplace=True)

final_dna_rna_mirna_df = pd.merge(final_dna_rna_df, final_mirna_df, left_index=True, right_index=True)

del final_dna_rna_df, final_mirna_df  # release memory
gc.collect()
print("final_dna_rna_mirna_df: ", final_dna_rna_mirna_df.shape)



final_rppa_df = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/RPPA.csv", index_col=0)
print("final_rppa_df: ", final_rppa_df.shape)
final_rppa_df.index = map(lambda s: str(s)[:15], final_rppa_df.index)
final_rppa_df.columns = map(lambda s: s + '_rppa', final_rppa_df.columns)

final_rppa_df.fillna(final_rppa_df.mean(), inplace=True)

final_dna_rna_mirna_rppa_df = pd.merge(final_dna_rna_mirna_df, final_rppa_df, left_index=True, right_index=True)

del final_dna_rna_mirna_df, final_rppa_df  # release memory
gc.collect()
print("final_dna_rna_mirna_rppa_df: ", final_dna_rna_mirna_rppa_df.shape)


# methy merge
# methy_df = pd.read_csv(r"../tmp_dir/GDC_methy.csv", index_col=0)  # 5 min
# methy_df = pd.read_csv(r"../data/GDC_methy_pf.csv", index_col=0)
methy_df = pd.read_csv('/home/SENSETIME/chenfeiyang/data1/data/newfea_GDC_methy.csv', index_col=0)
print("methy_df: ", methy_df.shape)
methy_df.index = map(lambda s: str(s)[:15], methy_df.index)
methy_df.columns = map(lambda s: s + '_methy', methy_df.columns)
# methy_df.fillna(0,inplace=True)
# fill na
# for col_name in list(methy_df.columns[final_rna_df.isnull().sum() > 0]):
#     mean_val = methy_df[col_name].mean()
#     methy_df[col_name].fillna(mean_val, inplace=True)
methy_df.fillna(methy_df.mean(), inplace=True)

final_dna_rna_mirna_rppa_methy_df = pd.merge(final_dna_rna_mirna_rppa_df, methy_df, left_index=True, right_index=True)
print("final_dna_rna_mirna_rppa_methy_df: ", final_dna_rna_mirna_rppa_methy_df.shape)

del final_dna_rna_mirna_rppa_df, methy_df
gc.collect()
# del tmp_methy_df

# In[11]:


final_cnv_df = pd.read_csv("/home/SENSETIME/chenfeiyang/data1/data/CNV.csv", index_col=0)
print("final_cnv_df: ", final_cnv_df.shape)
final_cnv_df.columns = map(lambda s: s + '_cnv', final_cnv_df.columns)

final_dna_rna_mirna_rppa_methy_cnv_df = pd.merge(final_dna_rna_mirna_rppa_methy_df, final_cnv_df, left_index=True, right_index=True)
print("final_dna_rna_mirna_rppa_methy_cnv_df: ", final_dna_rna_mirna_rppa_methy_cnv_df.shape)

del final_dna_rna_mirna_rppa_methy_df, final_cnv_df
gc.collect()
# final_dna_rna_methy_df.to_csv('../processed_dir/gdc_all_data.csv')
final_dna_rna_mirna_rppa_methy_cnv_df.to_csv('/home/SENSETIME/chenfeiyang/data1/data/gdc_all_data.csv')
# final_dna_rna_methy_df = pd.read_csv(r"../data/gdc_all_data.csv", index_col=0)
# In[12]:

print("Done")

del final_dna_rna_mirna_rppa_methy_cnv_df
gc.collect()



final_dna_rna_mirna_rppa_methy_df = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/df_all_new_new.csv", index_col=0)

print("final_dna_rna_mirna_rppa_methy_df: ", final_dna_rna_mirna_rppa_methy_df.shape)


# ## 数据划分
# train、valid、primary_test、transfer_test (6: 2: 2: N) <br>
# 原发癌的患者 按照每个类别 划分为训练、测试、验证，比例为 6：2：2， 所有转移患者为额外测试集
#
# train_ft, train_label, valid_ft, valid_label.......
#

# In[59]:


def train_test_split(record_num, split_param, mode='ratio', shuffle=False, category=None, extra_to='train'):
    '''
    :param record_num:
    :param split_param: The proportion or number of data divided
    :param mode: 'ratio' or 'numerical'
    :param shuffle:
    :param category: list record_num
    :return:  each category has the same ratio of train/valid/test
    '''

    if category is not None:
        set_of_category = set(category)
        set_of_category = sorted(set_of_category)
        # print(set_of_category)
        # print(set_of_category, count_num_of_category)

        # 每一种类别 对应的idx

        all_idx = np.arange(record_num)
        category = np.array(category)
        category_idx_dict = {}
        # print(len(category))
        # print('all c', type(category), category)
        for c in set_of_category:
            category_idx_dict[c] = all_idx[c == category]
        # 对每一种类别 分别进行划分
        train_index = []
        valid_index = []
        test_index = []

        for c in category_idx_dict:
            index_arr = category_idx_dict[c]
            sub_train_index, sub_valid_index, sub_test_index = arr_split(
                index_arr, split_param=split_param, mode=mode, shuffle=shuffle, extra_to=extra_to)
            # print(c, sub_train_index, sub_valid_index, sub_test_index)
            train_index.append(sub_train_index)
            valid_index.append(sub_valid_index)
            test_index.append(sub_test_index)
        train_index = np.concatenate(train_index, axis=0)
        valid_index = np.concatenate(valid_index, axis=0)
        test_index = np.concatenate(test_index, axis=0)
        # print('train_index = ', train_index)
    else:
        index_arr = np.arange(record_num)
        # print('index_arr =')
        train_index, valid_index, test_index = arr_split(
            index_arr, split_param=split_param, mode=mode, shuffle=shuffle, extra_to=extra_to)

    return train_index, valid_index, test_index


def arr_split(raw_arr, split_param, mode='ratio', shuffle=False, extra_to='train'):
    '''
    :param raw_arr:  index np arr
    :param split_param:
    :param mode:
    :param shuffle:
    :return:  train/valid/test index arr
    '''
    record_num = raw_arr.shape[0]
    if mode == 'ratio' and sum(split_param) <= 1:
        split_param = np.array(split_param)
        train_size, valid_size, test_size = map(int, np.floor(split_param * record_num))
        extra_size = record_num - train_size - valid_size - test_size
        # print(train_size, valid_size, test_size, extra_size, extra_to)
        if extra_to == 'train':
            train_size += extra_size
        if extra_to == 'valid':
            valid_size += extra_size
        if extra_to == 'test':
            test_size += extra_size
        # print(train_size, valid_size, test_size, extra_size)
    elif mode == 'numerical' or sum(split_param) > 1:
        split_param = np.array(split_param)
        # print(np.sum(split_param), node_size)
        assert np.sum(split_param) <= record_num
        train_size, valid_size, test_size = split_param

    node_index = np.arange(record_num)
    if shuffle is True:
        np.random.shuffle(node_index)

    train_pos = node_index[0:train_size]
    valid_pos = node_index[train_size: valid_size + train_size]
    test_pos = node_index[valid_size + train_size: valid_size + train_size + test_size]

    return raw_arr[train_pos], raw_arr[valid_pos], raw_arr[test_pos]


# In[60]:


# final_dna_rna_methy_df = pd.read_csv('../data/gdc_all_data.csv', index_col=0)

# In[61]:


# dna_rna_methy_y[dna_rna_methy_y["is_primary"] == "transfer"].shape
np.random.seed(666)
random.seed(666)
# label str to num
enc = LabelEncoder()
num_label = enc.fit_transform(final_dna_rna_mirna_rppa_methy_df['cancer_type'])
final_dna_rna_mirna_rppa_methy_df['num_label'] = num_label

primary_df = final_dna_rna_mirna_rppa_methy_df[final_dna_rna_mirna_rppa_methy_df["is_primary"] == "primary"].copy()
transfer_df = final_dna_rna_mirna_rppa_methy_df[final_dna_rna_mirna_rppa_methy_df["is_primary"] == "transfer"].copy()

primary_train_idx, primary_valid_idx, primary_test_idx = train_test_split(
    record_num=primary_df.shape[0],
    split_param=[0.6, 0.2, 0.2],
    mode='ratio',
    shuffle=True,
    category=primary_df['cancer_type'],
    extra_to='train'
)

print(primary_df.shape, transfer_df.shape, final_dna_rna_mirna_rppa_methy_df.shape)
print(primary_train_idx.shape, primary_valid_idx.shape, primary_test_idx.shape)

split_type = np.zeros(primary_df.shape[0])
split_type = split_type.astype(np.str)

split_type[primary_train_idx] = 'primary_train'
split_type[primary_valid_idx] = 'primary_valid'
split_type[primary_test_idx] = 'primary_test'
primary_df['split_type'] = split_type

primary_df.to_csv('/home/SENSETIME/chenfeiyang/data1/data/gdc_primary.csv')
transfer_df.to_csv('/home/SENSETIME/chenfeiyang/data1/data/gdc_transfer.csv')

df_all_new = pd.concat([primary_df,transfer_df])
df_all_new.to_csv('/home/SENSETIME/chenfeiyang/data1/data/df_all_new.csv')

print("Done")
# In[63]:


label_df = pd.DataFrame()
all_sample_idx = np.concatenate([primary_df.index, transfer_df.index], axis=0)
all_sample_cancer_type = np.concatenate([primary_df['cancer_type'], transfer_df['cancer_type']], axis=0)
all_sample_num_label = np.concatenate([primary_df['num_label'], transfer_df['num_label']], axis=0)

transfer_split_type = np.zeros(transfer_df.shape[0]).astype(np.str)
transfer_split_type.fill('transfer_test')
all_sample_split_type = np.concatenate([primary_df['split_type'], transfer_split_type], axis=0)

label_df['sample_idx'] = all_sample_idx
label_df['cancer_type'] = all_sample_cancer_type
label_df['split_type'] = all_sample_split_type
label_df['num_label'] = all_sample_num_label

label_df.to_csv('/home/SENSETIME/chenfeiyang/data1/data/gdc_label_info.csv')

print(label_df.shape)
