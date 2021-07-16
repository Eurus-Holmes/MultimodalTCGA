import pandas as pd
import numpy as np
import sys
import os
# ## DNA 文件处理
#
# 原始 DNA-SNV 的文件每个 item 为 (Sample_ID, gene, dna_vaf....)， 记录了某个患者基因的一次突变<br> 
# 1. 初步筛选，按filter和effect列进行筛选 
# 2. 统计列表中存在的所有 gene 
# 3. 根据 ID 合并存在的基因突变特征， 若未突变用0填充 
# 4. 当同一 ID， 同一 gene 有多个突变记录时，保留 dna_vaf 最高的值，dna_vaf的值进行cutoff=0.05，若dna_vaf大于0.05取1,否则取0 
#
# 数据保存为 DNA_VAF_GDC.csv 文件 (10154, 19655)

# In[4]:

##pf
# file_path = "GDC-PANCAN.mutect2_snv.tsv"
## yw
file_path = "/home/SENSETIME/chenfeiyang/data1/GDC_data/GDC-PANCAN.mutect2_snv.tsv"
data = pd.read_csv(file_path, sep="\t")  # 3175929 rows × 11 columns
# print(data.groupby("Sample_ID"))
# data
data = data[data['filter'] == 'PASS']
mask = data[['effect']].apply(
    lambda x: x.str.contains(
        'inframe|frameshift|missense|splice',
        regex=True
    )
).any(axis=1)
data = data[mask]
# all gene
gene_names = list(sorted(set(data['gene'])))
# print(len(gene_names))
gene_map_idx = {}
for idx in range(len(gene_names)):
    name = gene_names[idx]
    gene_map_idx[name] = idx

sample_id_list = []
sample_dna_snv_mat = []
idx = 0
for gp in data.groupby("Sample_ID"):  # 30mins

    sample_id_list.append(gp[0])  # TCGA-02-0003-01A
    df = gp[1]
    #     print(df.columns)
    #     ['Sample_ID', 'gene', 'chrom', 'start', 'end', 'ref', 'alt',
    #     Amino_Acid_Change', 'effect', 'filter', 'dna_vaf']

    df = df.sort_values(by=["gene", "dna_vaf"], ascending=False)[["gene", "dna_vaf"]]
    # 按照基因名排序
    dup_df = df.drop_duplicates(subset="gene")
    # 有多个表达量时， 默认选取第一个

    # sample df ->  num_array
    sample_dna_snv = np.zeros(len(gene_names))
    for idx, row in dup_df.iterrows():
        gene_name = row[0]
        dna_val = row[1]
        sample_dna_snv[gene_map_idx[gene_name]] = 1 if dna_val > 0.05 else 0
    sample_dna_snv_mat.append(sample_dna_snv)
    #     break
    if len(sample_dna_snv_mat) % 2000 == 0:
        print(len(sample_dna_snv_mat))

dna_snv_df = pd.DataFrame(sample_dna_snv_mat, columns=gene_names)
dna_snv_df['sample_id'] = sample_id_list
dna_snv_df = dna_snv_df.set_index('sample_id')
print(dna_snv_df.shape)
# (10154, 19655)
dna_snv_df.to_csv(r"/home/SENSETIME/chenfeiyang/data1/data/DNA_VAF_GDC.csv")
