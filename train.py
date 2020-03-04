#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:07:47 2020

@author: yki
"""
import datetime
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.nn as nn
import numpy as np
from libs.model import MLP
from libs.model import cornea
parser = OptionParser()
parser.add_option("--TRAIN_DIR",
                  dest="TRAIN_DIR", default='./dataset/cornea_4_train.csv',
                  help="training_file")

parser.add_option("--TEST_DIR",
                  dest="TEST_DIR", default='./dataset/cornea_4_test.csv',
                  help="test_file")

parser.add_option("--model_checkpoint_dir",
                  dest="out_dir", default='',
                  help="model_saving_dir")
parser.add_option("--epoch",
                  dest="epoch", default=100,
                  help="how many epoches in training")
parser.add_option("--times",
                  dest="times", default=4,
                  help="how many times of training")
#parser.add_option("--cropping_step",
#                  dest="cropping_step", default=2,
#                  help="the step of cropping full-sky CMB map when generate the dataset"
#                  +"if you want to generate a larger dataset , you can make this number smaller"
#                  )
options, args = parser.parse_args()
train_df = pd.read_csv(options.TRAIN_DIR) # cornea_4_train.csv for no diabetes; cornea_noage_noudva_2_train.csv for with diabetes
test_df = pd.read_csv(options.TEST_DIR)
X_train, X_valid, y_train, y_valid = \
    train_test_split(train_df.iloc[:, 0:29], train_df['label']-1, test_size=1/5, random_state=42)

X_test = test_df.iloc[:,0:29]
y_test = test_df['label']-1

train_dataset = cornea(X=X_train, y=y_train, transform=None)
valid_dataset = cornea(X=X_valid, y=y_valid, transform=None)
test_dataset = cornea(X=X_test, y=y_test, transform=None)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

print('train factors size : ', X_train.shape)
print('train labels size : ', y_train.shape)
print('valid factors size : ', X_valid.shape)
print('valid labels size : ', y_valid.shape)
print('test data size : ', test_df.shape)
runs = int(options.times)
epochs =int(options.epoch)
train_loss = []
val_loss = []
val_acc = []
test_acc = []
for run in range(runs):
    print('***** Run {} *****'.format(run))
  
    model = MLP()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []

    for epoch in range(epochs):
        model.train()
    
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
        
            optimizer.zero_grad()
        
            outputs, _ = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            train_losses.append(loss.item())
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                outputs, _ = model(images)
                loss = loss_fn(outputs, labels)
                valid_losses.append(loss.item())
            
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
    
        accuracy = 100*correct/total
        valid_acc_list.append(accuracy)
        if epoch%10==0:
          print('epoch: {}, train loss: {:.4f}, valid loss: {:.4f}, valid acc: {:.2f}%'\
              .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))
      
    train_loss.append(np.array(mean_train_losses))
    val_loss.append(np.array(mean_valid_losses))
    val_acc.append(np.array(valid_acc_list))
    
    
torch.save(model.state_dict(),options.out_dir+datetime.datetime.now().strftime('%d %H:%M')+"mlp_params.pt")