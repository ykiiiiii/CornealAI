#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:43:46 2020

@author: yki
"""
from torch.utils.data import DataLoader
from optparse import OptionParser
from libs.model import * 
import torch
import pandas as pd
from datetime import datetime
from dateutil import tz

parser = OptionParser()
parser.add_option("--TEST_DIR",
                  dest="TEST_DIR", default='./dataset/cornea_4_test.csv',
                  help="test_file")
parser.add_option("--weight",
                  dest="weight", default=None,
                  help='pre_trained')
options, args = parser.parse_args()
weight = options.weight#weight name
model = MLP()
model.load_state_dict(torch.load(weight))

test = options.TEST_DIR#test data  directory
test_df = pd.read_csv(test)
X_test = test_df.iloc[:,0:29]
y_test = test_df['label']-1
test_dataset = cornea(X=X_test, y=y_test, transform=None)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
output = []
for i,(data,labels) in enumerate(test_loader):
  out,outs = model(data)
  _, pred_label = torch.max(out.data,1)
  output.append(int(pred_label)+1)
test_df['predicted labels']=output

tz_sh = tz.gettz('Australia/Sydney')
# Shanghai timezone
now_sh = datetime.now(tz=tz_sh).strftime('%m%d %H:%M')
test_df.to_csv('testdata predicted at'+now_sh+'.csv')