 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:28:07 2021

@author: cjove
"""
import data_extraction as de
import dataset_generation_step as dgs
import dataset_generation_s2s as dgs2s
import pandas as pd




test = pd.read_csv("C:/Users/cjove/Desktop/TFM/Dataset/AB156/AB156/Raw/AB156_Circuit_001_raw.csv")
names = list(test.columns)
dir_dataset = 'C:/Users/cjove/Desktop/TFM/Dataset/'

data = de.data_extraction(dir_dataset, names)


emg_filt = data[0]
acc_filt = data[1]
gon_filt = data[2]

acc_names = data[3]
muscle_names = data[4]
index_steps = data[5]
"""
Obtención del dataset step
"""


imu_features = dgs.acc_features_step(acc_filt, acc_names, index_steps)

emg_features = dgs.emg_features_step(emg_filt, muscle_names, index_steps, 
                                 data_gy_d = acc_filt['Right_Shank_Gy'],
                                 data_gy_i = acc_filt['Left_Shank_Gy']) 


lab_imu = dgs.create_labels_imu(acc_names, dgs.featurename_imu )

lab_emg = dgs.create_labels_emg(muscle_names, dgs.featurename_emg)

              
dataframe_imu_step = dgs.create_dataframe_step(imu_features, lab_imu, info = 0)   
dataframe_emg_step = dgs.create_dataframe_step(emg_features, lab_emg, 2)       



"""
Obtención del dataset transición sedestación a bipedestación
"""

imu_features_s2s = dgs2s.acc_features_sit2stand(acc_filt, acc_names)
emg_features_s2s = dgs2s.emg_features_sit2stand(emg_filt, muscle_names,acc_filt)

imu_s2s_dataframe = dgs2s.create_dataframe_s2s(imu_features_s2s, lab_imu, 0)
emg_s2s_dataframe = dgs2s.create_dataframe_s2s(emg_features_s2s, lab_emg, 1)

print('¡Proceso Completado!')





























































