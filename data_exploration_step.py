# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:11:10 2021

@author: cjove
"""
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import scatter_matrix



names_emg_step = steps_emg_ordered.columns
names_imu_step = steps_imu_ordered.columns

shape_emg_step = steps_emg_ordered.shape
shape_imu_step = steps_imu_ordered.shape

steps_emg_ordered.info()
steps_imu_ordered.info()


"""Descripción del dataset"""

description_emg_step = steps_emg_ordered.describe()
description_imu_step = steps_imu_ordered.describe()
description_y_step = y_step.describe()

"""Correlaciones"""

corrMatrix_imu_step = steps_imu_ordered.corr()
corrMatrix_emg_step = steps_emg_ordered.corr()


"""Heatmaps"""
sn.heatmap(corrMatrix_imu_step, annot=False)
plt.show()

sn.heatmap(corrMatrix_emg_step, annot=False)
plt.show()

"""Examinamos la distribución de las variables"""

"""Boxplot"""
length_emg = np.arange(0,shape_emg_step[1],10)
length_imu = np.arange(0,shape_imu_step[1],10)

for i in length_emg:
    plt.boxplot(steps_emg_ordered[names_emg_step[i:i+10]], labels=names_emg_step[i:i+10])
    plt.show()
    plt.violinplot(steps_emg_ordered[names_emg_step[i:i+10]])
    plt.show()

for i in length_imu:
    plt.boxplot(steps_imu_ordered[names_imu_step[i:i+10]], labels=names_imu_step[i:i+10])
    plt.show()
    plt.violinplot(steps_imu_ordered[names_imu_step[i:i+10]])
    plt.show()
    

"""Exploramos el efecto de escalar los datos"""

scaler = StandardScaler()
scaler.fit(steps_imu_ordered)
steps_imu_scaled = scaler.transform(steps_imu_ordered)
steps_imu_scaled = pd.DataFrame(steps_imu_scaled, columns = names_imu_step)
scaler.fit(steps_emg_ordered)
steps_emg_scaled = scaler.transform(steps_emg_ordered)
steps_emg_scaled = pd.DataFrame(steps_emg_scaled, columns = names_emg_step)

for i in length_emg:
    plt.boxplot(steps_emg_scaled[names_emg_step[i:i+10]], labels=names_emg_step[i:i+10])
    plt.show()
    plt.violinplot(steps_emg_scaled[names_emg_step[i:i+10]])
    plt.show()

for i in length_imu:
    plt.boxplot(steps_imu_scaled[names_imu_step[i:i+10]], labels=names_imu_step[i:i+10])
    plt.show()
    plt.violinplot(steps_imu_scaled[names_imu_step[i:i+10]])
    plt.show()

    
"""
Fuente:
https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
"""
#1. Isolation Forest

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.11)
yhat_emg = iso.fit_predict(steps_emg_scaled)
yhat_imu = iso.fit_predict(steps_imu_scaled)

mask_emg = yhat_emg != -1
mask_imu = yhat_imu != -1

print(yhat_emg[mask_emg].shape, yhat_emg.shape)
print(yhat_imu[mask_imu].shape, yhat_imu.shape)


#2.Minimum Covariance Determinant

from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(contamination=0.01)

yhat_emg_ee = ee.fit_predict(steps_emg_scaled)
yhat_imu_ee = ee.fit_predict(steps_imu_scaled)

mask_emg_ee = yhat_emg_ee != -1
mask_imu_ee = yhat_imu_ee != -1

print(yhat_emg_ee[mask_emg_ee].shape, yhat_emg_ee.shape)
print(yhat_imu_ee[mask_imu_ee].shape, yhat_imu_ee.shape)

#3.OneClassSVM

from sklearn.svm import OneClassSVM

oc = OneClassSVM(nu=0.01)
yhat_emg_oc = oc.fit_predict(steps_emg_scaled)
yhat_imu_oc = oc.fit_predict(steps_imu_scaled)

mask_emg_oc = yhat_emg_oc != -1
mask_imu_oc = yhat_imu_oc != -1

print(yhat_emg_ee[mask_emg_oc].shape, yhat_emg_oc.shape)
print(yhat_imu_ee[mask_imu_oc].shape, yhat_imu_oc.shape)


list_values = np.arange(0.01,1,0.1)

isof_step = []
ee_step = []
oc_step = []

for i in list_values:
    iso = IsolationForest(contamination= i)
    yhat_emg = iso.fit_predict(steps_emg_scaled)
    yhat_imu = iso.fit_predict(steps_imu_scaled)
    
    mask_emg = yhat_emg != -1
    mask_imu = yhat_imu != -1
    
    emg = yhat_emg[mask_emg].shape
    imu = yhat_imu[mask_imu].shape
    
    isof_step.append({'EMG':emg,'IMU':imu})


    
    ee = EllipticEnvelope(contamination=i)
    
    yhat_emg_ee = ee.fit_predict(steps_emg_scaled)
    yhat_imu_ee = ee.fit_predict(steps_imu_scaled)
    
    mask_emg_ee = yhat_emg_ee != -1
    mask_imu_ee = yhat_imu_ee != -1
    
    emg = yhat_emg_ee[mask_emg_ee].shape
    imu = yhat_imu_ee[mask_imu_ee].shape
    
    ee_step.append({'EMG':emg,'IMU':imu})

    
    oc = OneClassSVM(nu=i)
    yhat_emg_oc = oc.fit_predict(steps_emg_scaled)
    yhat_imu_oc = oc.fit_predict(steps_imu_scaled)
    
    mask_emg_oc = yhat_emg_oc != -1
    mask_imu_oc = yhat_imu_oc != -1
    
    emg = yhat_emg_ee[mask_emg_oc].shape
    imu = yhat_imu_ee[mask_imu_oc].shape

    oc_step.append({'EMG':emg,'IMU':imu})

#out_iso_imu_step = out_iso_imu_step.drop(['IsoForest'], axis=1)
#out_iso_emg_step = out_iso_emg_step.drop(['IsoForest'], axis=1)     

import pandas as pd
outliers_step = pd.DataFrame({
    'Isolation Forest': isof_step,
    'Minimum Covariance Determinant': ee_step,
    'OneClassSVM': oc_step}, index = list_values
    )

"""
Generación del dataset sin outliers
"""
steps_imu_ordered_out = steps_imu_ordered.copy()
steps_emg_ordered_out = steps_emg_ordered.copy()

steps_emg_ordered_out['IsoForest'] = yhat_emg
steps_imu_ordered_out['IsoForest'] = yhat_emg
y_step_out = y_step.copy()
y_step_out['IsoForest'] = yhat_emg


out_iso_imu_step = steps_imu_ordered_out.loc[steps_imu_ordered_out['IsoForest'] ==1]
out_iso_emg_step = steps_emg_ordered_out.loc[steps_emg_ordered_out['IsoForest'] ==1]
y_step_out = y_step_out.loc[y_step_out['IsoForest'] ==1]
out_iso_imu_step = out_iso_imu_step.drop(['IsoForest'], axis=1)
out_iso_emg_step = out_iso_emg_step.drop(['IsoForest'], axis=1)                                             
y_step_out = y_step_out.drop(['IsoForest'],axis =1)











