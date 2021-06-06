# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:07:50 2021

@author: cjove
"""

import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import scatter_matrix





names_emg_s2s = emg_s2s_ordered.columns
names_imu_s2s = imu_s2s_ordered.columns
#names_imu_s2s = names_imu_s2s[72:]

shape_emg_s2s = emg_s2s_ordered.shape
shape_imu_s2s = imu_s2s_ordered.shape

emg_s2s_ordered.info()
imu_s2s_ordered.info()


"""Descripción del dataset"""

description_emg_s2s = emg_s2s_ordered.describe()
description_imu_s2s = imu_s2s_ordered[names_imu_s2s[72:]].describe()
description_y = y.describe()
plt.boxplot(y, labels = y.columns)

"""Correlaciones"""

corrMatrix_imu_s2s = imu_s2s_ordered[names_imu_s2s[72:]].corr()
corrMatrix_emg_s2s = emg_s2s_ordered.corr()


"""Heatmaps"""
sn.heatmap(corrMatrix_imu_s2s, annot=False)
plt.show()

sn.heatmap(corrMatrix_emg_s2s, annot=False)
plt.show()

"""Examinamos la distribución de las variables"""

"""Boxplot"""
length_emg = np.arange(0,shape_emg_s2s[1],10)
length_imu = np.arange(0,shape_imu_s2s[1],10)

for i in length_emg:
    plt.boxplot(emg_s2s_ordered[names_emg_s2s[i:i+10]], labels=names_emg_s2s[i:i+10])
    plt.show()
    plt.violinplot(emg_s2s_ordered[names_emg_s2s[i:i+10]])
    plt.show()

for i in length_imu:
    plt.boxplot(imu_s2s_ordered[names_imu_s2s[i:i+10]], labels=names_imu_s2s[i:i+10])
    plt.show()
    plt.violinplot(imu_s2s_ordered[names_imu_s2s[i:i+10]])
    plt.show()
    

"""Exploramos el efecto de escalar los datos"""

scaler = StandardScaler()
scaler.fit(imu_s2s_ordered)
s2s_imu_scaled = scaler.transform(imu_s2s_ordered)
s2s_imu_scaled = pd.DataFrame(s2s_imu_scaled, columns = names_imu_s2s)
scaler.fit(emg_s2s_ordered)
s2s_emg_scaled = scaler.transform(emg_s2s_ordered)
s2s_emg_scaled = pd.DataFrame(s2s_emg_scaled, columns = names_emg_s2s)

for i in length_emg:
    plt.boxplot(s2s_emg_scaled[names_emg_s2s[i:i+10]], labels=names_emg_s2s[i:i+10])
    plt.show()
    plt.violinplot(s2s_emg_scaled[names_emg_s2s[i:i+10]])
    plt.show()

for i in length_imu:
    plt.boxplot(s2s_imu_scaled[names_imu_s2s[i:i+10]], labels=names_imu_s2s[i:i+10])
    plt.show()
    plt.violinplot(s2s_imu_scaled[names_imu_s2s[i:i+10]])
    plt.show()

    
"""
Fuente:
https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
"""
#1. Isolation Forest

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.11)
s2s_yhat_emg = iso.fit_predict(s2s_emg_scaled)
s2s_yhat_imu = iso.fit_predict(s2s_imu_scaled)

s2s_mask_emg = s2s_yhat_emg != -1
s2s_mask_imu = s2s_yhat_imu != -1

print(s2s_yhat_emg[s2s_mask_emg].shape, s2s_yhat_emg.shape)
print(s2s_yhat_imu[s2s_mask_imu].shape, s2s_yhat_imu.shape)


#2.Minimum Covariance Determinant

from sklearn.covariance import EllipticEnvelope

s2s_ee = EllipticEnvelope(contamination=0.01)

s2s_yhat_emg_ee = s2s_ee.fit_predict(s2s_emg_scaled)
s2s_yhat_imu_ee = s2s_ee.fit_predict(s2s_imu_scaled)

s2s_mask_emg_ee = s2s_yhat_emg_ee != -1
s2s_mask_imu_ee = s2s_yhat_imu_ee != -1

print(s2s_yhat_emg_ee[s2s_mask_emg_ee].shape, s2s_yhat_emg_ee.shape)
print(s2s_yhat_imu_ee[s2s_mask_imu_ee].shape, s2s_yhat_imu_ee.shape)

s2s_imu_scaled['EE'] = s2s_mask_imu_ee
s2s_emg_scaled['EE'] = s2s_mask_emg_ee

#3.OneClassSVM

from sklearn.svm import OneClassSVM

s2s_oc = OneClassSVM(nu=0.01)
s2s_yhat_emg_oc = s2s_oc.fit_predict(s2s_emg_scaled)
s2s_yhat_imu_oc = s2s_oc.fit_predict(s2s_imu_scaled)

s2s_mask_emg_oc = s2s_yhat_emg_oc != -1
s2s_mask_imu_oc = s2s_yhat_imu_oc != -1

print(s2s_yhat_emg_oc[s2s_mask_emg_oc].shape, s2s_yhat_emg_oc.shape)
print(s2s_yhat_imu_oc[s2s_mask_imu_oc].shape, s2s_yhat_imu_oc.shape)

s2s_imu_scaled['OC'] = s2s_mask_imu_oc
s2s_emg_scaled['OC'] = s2s_mask_emg_oc



length_emg = np.arange(0,out_iso_emg[1],10)
length_imu = np.arange(0,shape_imu_s2s[1],10)


for i in length_emg:
    plt.boxplot(out_iso_emg[names_emg_s2s[i:i+10]], labels=names_emg_s2s[i:i+10])
    plt.show()
    plt.violinplot(out_iso_emg[names_emg_s2s[i:i+10]])
    plt.show()
  
import numpy as np

list_values = np.arange(0.01,1,0.1)

isof = []
ee = []
oc = []

for i in list_values:
    iso = IsolationForest(contamination=i)
    s2s_yhat_emg = iso.fit_predict(s2s_emg_scaled)
    s2s_yhat_imu = iso.fit_predict(s2s_imu_scaled)
    
    s2s_mask_emg = s2s_yhat_emg != -1
    s2s_mask_imu = s2s_yhat_imu != -1
    
    emg = s2s_yhat_emg[s2s_mask_emg].shape
    imu = s2s_yhat_imu[s2s_mask_imu].shape
    
    isof.append({'EMG':emg,'IMU':imu})
    
    s2s_ee = EllipticEnvelope(contamination= i)

    s2s_yhat_emg_ee = s2s_ee.fit_predict(s2s_emg_scaled)
    s2s_yhat_imu_ee = s2s_ee.fit_predict(s2s_imu_scaled)

    s2s_mask_emg_ee = s2s_yhat_emg_ee != -1
    s2s_mask_imu_ee = s2s_yhat_imu_ee != -1

    emg = s2s_yhat_emg_ee[s2s_mask_emg_ee].shape
    imu = s2s_yhat_imu_ee[s2s_mask_imu_ee].shape
    
    ee.append({'EMG':emg,'IMU':imu})
    
    s2s_oc = OneClassSVM(nu=i)
    s2s_yhat_emg_oc = s2s_oc.fit_predict(s2s_emg_scaled)
    s2s_yhat_imu_oc = s2s_oc.fit_predict(s2s_imu_scaled)
    
    s2s_mask_emg_oc = s2s_yhat_emg_oc != -1
    s2s_mask_imu_oc = s2s_yhat_imu_oc != -1
    
    emg = s2s_yhat_emg_oc[s2s_mask_emg_oc].shape
    imu = s2s_yhat_imu_oc[s2s_mask_imu_oc].shape
    
    oc.append({'EMG':emg,'IMU':imu})

import pandas as pd
outliers = pd.DataFrame({
    'Isolation Forest': isof,
    'Minimum Covariance Determinant': ee,
    'OneClassSVM': oc}, index = list_values
    )




"""
Generación del dataset sin outliers
"""
imu_s2s_ordered_out = imu_s2s_ordered.copy()
emg_s2s_ordered_out = emg_s2s_ordered.copy()
imu_s2s_ordered_out['IsoForest'] = s2s_yhat_emg
emg_s2s_ordered_out['IsoForest'] = s2s_yhat_emg
y_out = y.copy()
y_out['IsoForest'] = s2s_yhat_emg

out_iso_imu_s2s = imu_s2s_ordered_out.loc[imu_s2s_ordered_out['IsoForest'] ==1]
out_iso_emg_s2s = emg_s2s_ordered_out.loc[emg_s2s_ordered_out['IsoForest'] ==1]
y_out = y_out.loc[y_out['IsoForest'] ==1]


out_iso_imu_s2s = out_iso_imu_s2s.drop(['IsoForest'], axis=1)
out_iso_emg_s2s = out_iso_emg_s2s.drop(['IsoForest'], axis=1)                                             
y_out = y_out.drop(['IsoForest'], axis=1)

























#This url for feature selection
#https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/

#Pearson’s correlation coefficient (linear).
#Spearman’s rank coefficient (nonlinear)

#Let's try and see if this works with multioutput regressions

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X = steps_emg_scaled[names_emg_step[0:30]]
y = steps_emg_scaled[names_emg_step[50:55]]
# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)

#No permite llevarlo a cabo

import model_creation_functions as mcf
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

def bucle_ridge(desired_list, imu_frag, emg_frag, y):
    importances_r = {}
    importances_l = {}
    importances_e = {}
    
    length = np.arange(0,len(desired_list),1)
    for i in length:
        labels = desired_list[i]
        print('Empezando con el grupo:' +str(labels))
        X,y1 = mcf.combination(labels,imu_frag, emg_frag,y)
        print(X.columns)
        cols = X.columns
        max_features = int(len(cols))
        cols_y = len(y1.columns)
        print(max_features)
        train_X, test_X,  train_y, test_y = mcf.data_preparation(X,y1)
        ridge = Ridge()
        lasso = Lasso()
        elastic = ElasticNet()
        ridge.fit(train_X, train_y) 
        lasso.fit(train_X, train_y) 
        elastic.fit(train_X, train_y) 
        importance_r = ridge.coef_
        importance_l = lasso.coef_
        importance_e = elastic.coef_
        
        importances_r[labels] = importance_r
        importances_l[labels] = importance_l
        importances_e[labels] = importance_e

    return {'Ridge':importances_r,'Lasso':importances_l,'Elastic': importances_e}

selection_step = bucle_ridge(desired_list, imu_frag_step, emg_frag_step, y_step)

possible_combinations_step = mcf.powerset(emg_elements_step)

desired_list = [("Shank",)+v for v in possible_combinations_step]


desired_list = [x for x in desired_list if len(x)==3]

#dataset_step = desired_list[15:17]
u = 1
for x in desired_list:
    ridge = selection_step['Ridge'][x][u,:]
    lasso = selection_step['Lasso'][x][u,:]
    elastic = selection_step['Elastic'][x][u,:]
    #for i,v in enumerate(ridge):
        #print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
    plt.title('Ridge' + str(x))
    plt.bar([x for x in range(len(ridge))], ridge)
    plt.show()
    
    plt.title('Lasso' + str(x))
    plt.bar([x for x in range(len(lasso))], lasso)
    plt.show()
    
    plt.title('Elastic' + str(x))
    plt.bar([x for x in range(len(elastic))], elastic)
    plt.show()




corrMatrix_rms_step = y_step.corr()

# I'm missing here the exploration between imu and emg

# Heatmaps
sn.heatmap(corrMatrix_rms_step , annot=False)
plt.show()

mdf = steps.iloc[:, 56:63]
corrMatrix_mdf_step = mdf.corr()
sn.heatmap(corrMatrix_mdf_step , annot=False)
plt.show()

rms_mdf = pd.concat([y_step,mdf],axis = 1)
corrMatrix_rmsmdf_step = rms_mdf.corr()
sn.heatmap(corrMatrix_rmsmdf_step , annot=False)
plt.show()
