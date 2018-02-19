#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:30:46 2018

@author: cbdd
"""
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os

###################################### Load model ##########
current_path = os.path.split(os.path.realpath(__file__))[0]
cf = joblib.load(current_path+'/CYP3A4-substrate.pkl')

###################################### Load descriptors ##########
fingerprint_content = pd.read_csv(current_path+'/des.csv').ix[:, 1:]
des_list = np.array(fingerprint_content)

###################################### Prediction ##########
y_predict_label = cf.predict(des_list)
y_predict_proba = cf.predict_proba(des_list)
print '#'*10+'Results labels'+'#'*10
print y_predict_label
print '#'*10+'Results probabilities'+'#'*10
print y_predict_proba