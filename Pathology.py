# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from skimage import io
import histomicstk as htk
import skimage.io
import skimage.measure
import skimage.color
Image.MAX_IMAGE_PIXELS = 100000000000
#cv2.OPENCV_IO_MAX_IMAGE_PIXELS=100000000000
#cv2.CV_IO_MAX_IMAGE_PIXELS = 100000000000
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '2000000000000')
print(os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'])
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# sys.exit()
#%%
#train_data = pd.read_csv(r"../data/dna_rna_methy_train_gdc.csv")
#test_data = pd.read_csv(r"../data/dna_rna_methy_test_gdc.csv")
##
#primary_data = pd.read_csv(r"../data/gdc_primary.csv",index_col=0)
#transfer_data = pd.read_csv(r"../data/gdc_transfer.csv",index_col=0)
##
#data = pd.concat([primary_data,transfer_data])
#image_file = os.listdir(r"../image_data")
###
#data["sample"] = data.index.str[:-1]
#df = pd.DataFrame()
#images = [i for i in image_file if i[:15] in data["sample"].values]
#df["image_name"] = np.array(images)
#df["sample"] = df["image_name"].apply(lambda x:x[:15])
#df.drop_duplicates(subset="sample",keep="first",inplace=True)
#
##
#df_all = pd.merge(df,data,how="inner",on="sample")
#df_all.to_csv("../data/df_all_image.csv")
# df_all = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/df_all_new.csv",index_col=0)
new = pd.read_csv(r"/home/SENSETIME/chenfeiyang/data1/data/df_all_new.csv", index_col=0, usecols=[0,1,2])

new_list = list(new['image_name'])

path = "/home/SENSETIME/chenfeiyang/data1/image_deconvolved"

filelist = os.listdir(path)
img_name = []
sample = []

for item in filelist:
    img_name.append(item)


res = []
for name in new_list:
    new_name = name+'.npy'
    if new_name not in img_name:
        print(new_name)
        res.append(name)

#transfer_df = df_all[df_all["is_primary"] == "transfer"]
#df_primary = df_all[df_all["is_primary"] != "transfer"]
#transfer_df[["image_name","cancer_type","is_primary"]].to_csv(r"../data/transfer_df.csv",index=False)
#df_primary[["image_name","cancer_type","is_primary"]].to_csv(r"../data/df_primary.csv",index=False)
#df_all.to_csv(r"../data/df_all.csv")




# train_data = df_all[df_all["split_type"]=="primary_train"]
# valid_data = df_all[df_all["split_type"]=="primary_valid"]
# test_data = df_all[df_all["split_type"]=="primary_test"]
# transfer_data = df_all[df_all["split_type"].isna()]
# test_data = pd.concat([test_data,transfer_data])



#train_data,test_data = train_test_split(df_primary,train_size=0.8,stratify=df_primary["cancer_type"],random_state=666)
#train_data,valid_data = train_test_split(train_data,train_size=0.8,stratify=train_data["cancer_type"],random_state=666)
#test_data = pd.concat([test_data,transfer_df])


#sys.exit()
#def read_image(data):
#    data_list,data_target = [],[]
#    i=0
#    for inx in data.index:
#        ct = data.loc[inx,"cancer_type"]
#        image_name = data.loc[inx,"image_name"]
#        image = cv2.imread(f"../image_data/{image_name}")
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
#        data_list.append(th1)
#        data_target.append(ct)
#        i+=1
#        if i%100==0:
#            print(i)
#    return data_list,data_target
#train_list,train_target = read_image(train_data)
#print("train success")
#valid_list,valid_target = read_image(valid_data)
#print("valid success")
#test_list,test_target = read_image(test_data)
#print("test success")




#%%
folder_path = "/home/SENSETIME/chenfeiyang/data1/image_data"
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_color_map:', stain_color_map, sep='\n')
stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null'] 
# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T
for image in res:
    image_path = os.path.join(folder_path,image)
    imInput = skimage.io.imread(image_path)[:, :, :3]
    if imInput.shape[0]>6000 and imInput.shape[1]>6000:
        imInput = cv2.resize(imInput, (6000, 6000), interpolation=cv2.INTER_CUBIC)

    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, W)
    np.save(f"/home/SENSETIME/chenfeiyang/data1/image_deconvolved2/{image}.npy",imDeconvolved.Stains)
#    test = np.load(f"../image_deconvolved/{image}.npy")
    print(image)
# sys.exit()
#%%

"""
class CNNDataset:
    def __init__(self, image_name, targets):
        self.image_name = image_name.values
        self.targets = targets
        self.folder_path = "../image_deconvolved"

    def __len__(self):
        return (self.targets.shape[0])
    
    def __getitem__(self, idx):
#        self.image = []
#        if len(idx)==1:
        img_name = self.image_name[idx]
        self.image_list = []
        image_size = 256
#        for name in img_name:
        
#            img_name = os.listdir(folder_path)
        name_path = os.path.join(self.folder_path,img_name+".npy")
        image = np.load(name_path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        result = np.zeros(image.shape,dtype=np.float32)
        cv2.normalize(image, result, 0, 1, cv2.NORM_MINMAX,dtype =cv2.CV_32F )
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        mean = image_size*[image_size*[mean]]
        std = image_size*[image_size*[std]]
        image = (result - mean) / std
#        self.image_list.append(image)
#        image = cv2.imread(f"../image_data/{img_name}")
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
#        self.image.append(th1)
#        
##            for img_name in self.image_name[idx]:
##                image = cv2.imread(f"../image_data/{img_name}")
##                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##                ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
##                self.image.append(th1)
#        self.sub_image_list,self.adj_mat_list = [],[]
#        for img in self.image:
#            sub_image,adj_mat = cut_image(img)
#            self.sub_image_list.append(sub_image)
#            self.adj_mat_list.append(adj_mat)
#        self.sub_image_list = np.array(self.sub_image_list)[0]
#        self.adj_mat_list = np.array(self.adj_mat_list)[0]
        dct = {
            'x' : torch.tensor(image, dtype=torch.float),
            
            'y' : torch.tensor(self.targets[idx], dtype=torch.float)
        }
        return dct


BATCH_SIZE = 32
num_workers = 8
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(train_data["cancer_type"])
#train_Y = le.transform(train_data["cancer_type"])
#valid_Y = le.transform(valid_data["cancer_type"])
#test_Y = le.transform(test_data["cancer_type"])
#train_data.reset_index(drop=True,inplace=True)
#valid_data.reset_index(drop=True,inplace=True)
#test_data.reset_index(drop=True,inplace=True)
train_dataset = CNNDataset(train_data["image_name"], train_data["num_label"].values)
valid_dataset = CNNDataset(valid_data["image_name"], valid_data["num_label"].values)
test_dataset = CNNDataset(test_data["image_name"], test_data["num_label"].values)
#print(train_dataset[0])
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,
                                           num_workers=num_workers,shuffle=True,pin_memory=False)

valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,
                                            num_workers=num_workers,shuffle=True,pin_memory=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                          num_workers=num_workers,shuffle=False,pin_memory=False)
#sys.exit()
#    if len(th1_sub)==0:
#        th1_sub = sub_tmp
#    else:
        
#        th1_sub = np.concatenate((th1_sub,sub_tmp),axis=0)

# from CNN_M import CNN
#from CNN import Net
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.")
else:
    print("CUDA is available!")
num_classes = train_data["cancer_type"].nunique()
model = CNN(32,num_classes)
if train_on_gpu:
    model.cuda()

import torch.optim as optim

# use crossentropyloss
criterion = nn.CrossEntropyLoss()
#criterion = LabelSmoothingCrossEntropy()
## lr=0.01
optimizer = optim.Adam(model.parameters(), lr=1e-4)

## train epoch
n_epochs = 1000
valid_loss_min = np.Inf # track change in validation loss
for epoch in range(1, n_epochs+1):
    
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    output_list = []
    target_list = []
    for dct in tqdm(train_loader):
#        print(data_target["data"],data_target["target"])
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data,target = dct["x"].cuda(), dct["y"].cuda()
        else:
            data,target = dct["x"],  dct["y"]
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted ouputs by passing inputs to
        
        output = model(data)
#        output = output.unsqueeze(0)
        if len(output_list)==0:
            output_list = output
            target_list = target
        else:
            output_list = torch.cat((output_list,output),axis=0)
            target_list = torch.cat((target_list,target),axis=0)
        # calculate the batch loss
        target=torch.tensor(target, dtype=torch.long)
        target = target.cuda()
        loss = criterion(output,target)

#        reg_loss = 0
#        for name, param in model.named_parameters():
#            if 'bias' not in name:
#                reg_loss += torch.norm(param, p=2)
#        loss += reg_loss*1e-2
                
        # backward pass: compute gradient of the loss with respect to
        loss.backward()
        # perform a single optimizaion step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
    acc = sum(torch.argmax(output_list,axis=1)==target_list).item()/len(target_list)
    print("train acc:",acc)
    torch.save(model.state_dict(),"model_train.pt")
    model.eval()
#    model.cpu()
#    torch.cuda.empty_cache()
#    model = torch.load("model_train.pt",map_location=lambda storage, loc: storage.cuda(0))
#    model.load("model_train.pt")
#    model.load_state_dict(torch.load("model_train.pt"))
#    model.eval()
    
#    model.cuda()
    with torch.no_grad():
        output_list = []
        target_list = []
        
        for dct in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data,target = dct["x"].cuda(), dct["y"].cuda()
            else:
                data,target = dct["x"],  dct["y"]
            # forward pass: compute predicted outputs by passing inputs to
    #        optimizer.zero_grad()
            output = model(data)
#            output = output.unsqueeze(0)
            if len(output_list)==0:
                output_list = output
                target_list = target
            else:
                output_list =torch.cat((output_list,output),axis=0)
                target_list =torch.cat((target_list,target),axis=0)
            target=torch.tensor(target, dtype=torch.long)
            target = target.cuda()
            # calculate the batch loss
            loss = criterion(output,target)
    #        optimizer.step()
            # update average validation loss
            valid_loss += loss.item()*data.size(0)
    acc = sum(torch.argmax(output_list,axis=1)==target_list).item()/len(target_list)
    print("valid acc:",acc)
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    print("Epoch:{} \tTraining Loss:{:.6f} \tValidation Loss:{:.6}".format(epoch,train_loss,valid_loss))

    if valid_loss <= valid_loss_min:
        print("Validation loss decreased ({:.6}--> {:.6f}). Saving model ...".format(
                valid_loss_min,
                valid_loss))
        valid_loss_min = valid_loss
torch.save(model.state_dict(),"model_cifar.pt")
# track test loss
sys.exit()
test_loss = 0.0
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

model.eval()

# iterate over test data

for data,target in tqdm(test_loader):
    # move tensors to GPU if CUDA is available

    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(len(target)):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1

# average test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))


for i in range(num_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

        100. * np.sum(class_correct) / np.sum(class_total),

        np.sum(class_correct), np.sum(class_total)))   

"""












