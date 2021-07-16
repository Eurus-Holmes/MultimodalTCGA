import pandas as pd
import numpy as np
import sys
import os
import mygene
from sklearn.preprocessing import LabelEncoder
import random
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

methy_df = pd.read_csv('/home/SENSETIME/chenfeiyang/data1/GDC_data/jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t', index_col=0)
methy_df = methy_df.T
methy_df = methy_df.fillna(0)
print(methy_df.shape)
# (9736,25979)
# (12039, 22601)
# methy_df["sample"] = methy_df.index.str[:-1]
methy_df["sample"] = methy_df.index.str[:15]
## ANOVA & the Bonferroni method &  Tukey’s honest significant difference post-hoc
# sample_label_df = pd.read_csv(r"/home/user/TCGA_DNA_methylation/Survival_SupplementalTable_S1_20171025_xena_sp",sep="\t")
# sample_label_df = sample_label_df[["sample","cancer type abbreviation"]]
# sample_label_df.rename(columns = {"cancer type abbreviation":"cancer_type"},inplace=True)
df_all = pd.read_csv('/home/SENSETIME/chenfeiyang/data1/GDC_data/df_all.csv', index_col=0)

methy_data = pd.merge(methy_df, df_all[["sample", "cancer_type", "split_type"]], how="inner", on="sample")
print(methy_data.shape)
# (7264, 22604)
print(methy_data.columns)
df = pd.DataFrame()
inx = 0
cg_name = []
p_value = []
f_value = []
for cg in methy_data.columns:
    if cg not in ["sample", "cancer_type", "split_type"]:
        #        for ct in methy_data["cancer_type"].unique():
        f, p = stats.f_oneway(*[s for idx, s in methy_data.groupby("cancer_type")[cg]])
        cg_name.append(cg)
        p_value.append(p)
        f_value.append(f)
        #        df.loc[inx,"cg_name"] = cg
        #        df.loc[inx,"p_value"] = p
        #        df.loc[inx,"f_value"] = f
        inx += 1
        if inx % 1000 == 0:
            print(inx, df.shape)
#            f, p = stats.f_oneway(methy_data[methy_data['cancer_type'] == ct][cg],
#                                  data[data['Archer'] == 'Jack'].Score,
#                                  data[data['Archer'] == 'Alex'].Score)
df["cg_name"] = np.array(cg_name)
df["p_value"] = np.array(p_value)
df["f_value"] = np.array(f_value)

from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

p_adjusted = multipletests(df["p_value"].values, alpha=0.01, method='bonferroni')
print("p_adjusted[0]: ", p_adjusted[0])

df_p = df[p_adjusted[0]]
print("df_p: ", df_p.shape)
## 23354
## (22569, 3)


p_true = [False]*len(df_p)
meandiff = [False]*len(df_p)
i = 0
for i, cg in enumerate(df_p["cg_name"]):
    res = pairwise_tukeyhsd(methy_data[cg], methy_data['cancer_type'],alpha=0.01)
    if sum(res.reject)>0:
        p_true[i] = True
    if sum(res.meandiffs>0.2)>0:
        meandiff[i] = True
    if i % 100 == 0:
        print(i)
    i += 1
df_p["p_true"] = np.array(p_true)
df_p["meandiff"] = np.array(meandiff)

dp_p_filter = df_p[df_p["p_true"] & df_p["meandiff"]]
print("dp_p_filter: ", dp_p_filter.shape)
# (11850, 5)
# mc = MultiComparison(methy_data['cg00000292'], methy_data['cancer_type'])
# mc = pairwise_tukeyhsd(methy_data['cg00000292'], methy_data['cancer_type'],alpha=0.01)
# result = mc.tukeyhsd(alpha=0.01)
# print(result)
train = methy_data[methy_data["split_type"] == "primary_train"]
test = methy_data[(methy_data["split_type"] == "primary_test") | (methy_data["split_type"].isna())]

features = [c for c in methy_data.columns if
            c not in ["sample", "cancer_type", "split_type"] and c in dp_p_filter["cg_name"].values]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

rf = RandomForestClassifier(n_estimators=100,
                            criterion="gini",
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.,
                            max_features="auto",
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.,
                            min_impurity_split=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=-1,
                            random_state=0,
                            verbose=0,
                            warm_start=False,
                            class_weight=None)

clf = rf.fit(train[features], train["cancer_type"])

pred = clf.predict(test[features])

acc = accuracy_score(test["cancer_type"], pred)

fea_imp = pd.DataFrame()
fea_imp["fea"] = np.array(features)
fea_imp["imp"] = np.array(clf.feature_importances_)
fea_imp.sort_values(by="imp", ascending=False, inplace=True)
df_fea_acc = pd.DataFrame()
fea_num = []
fea_acc = []
for n in range(len(features))[14990:15000]:
    if n % 1 == 0:
        fea_num.append(n)
        feas = fea_imp["fea"].values[:n]
        clf = rf.fit(train[feas], train["cancer_type"])
        pred = clf.predict(test[feas])
        acc = accuracy_score(test["cancer_type"], pred)
        fea_acc.append(acc)
        print(n, acc)
df_fea_acc["fea_num"] = np.array(fea_num)
df_fea_acc["fea_acc"] = np.array(fea_acc)

methy = methy_df[fea_imp["fea"].values[:15000]]
print(methy.shape)
## 22140
# (12039, 22140)
methy.to_csv(r"/home/SENSETIME/chenfeiyang/data1/data/newfea_GDC_methy.csv")
print("Done")
############################################### methy end ##########################################
del methy_df
