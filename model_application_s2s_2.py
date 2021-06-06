# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:42:29 2021

@author: cjove
"""

import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from itertools import chain, combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
#from implementation import imu_s2s_dataframe, emg_s2s_dataframe
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.stats import uniform as sp_rand
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import tensorflow as tf
from tensorflow import keras

import model_creation_functions as mcf 
"""
Generación de los modelos sin utilizar datos y param_grids para la función
RandomizedSearchCV
"""
ridge_reg = Ridge()
param_grid_ridge = {'alpha': sp_rand()}

lasso_reg = Lasso()
param_grid_lasso = {'alpha': sp_rand()}

elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
param_grid_elastic = {'alpha': sp_rand(),'l1_ratio':sp_rand()}

max_features = 2

tree_reg = DecisionTreeRegressor()
param_grid_decisiontree = {

    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': randint(1,max_features),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199),
    'max_depth': randint(4,20),
    'min_samples_leaf': randint(4,30)
}



random_forest = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 3)]
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]

max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
#bootstrap = False
bootstrap = [True, False]


param_grid_randomforest = {'n_estimators': n_estimators,
               'max_features':  ['auto', 'sqrt'],
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


extra_tree = ExtraTreesRegressor()
param_grid_extratree = {'n_estimators': n_estimators,
                        'criterion': ['mse', 'mae'],
                        'max_depth':max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}



param_grid_mlp = {
    "n_hidden":np.arange(1,10).tolist(),
    "n_neurons": np.arange(1,100).tolist(),
    "learning_rate": np.arange(3e-4,3e-2).tolist(),
    }

"""
param_grid_mlp = {
    "n_hidden":randint(1,100),
    "n_neurons": np.arange(1,100).tolist(),
    "learning_rate": np.arange(3e-4,3e-2).tolist(),
    }
"""

knreg = KNeighborsRegressor()
param_grid_knreg = {"n_neighbors" :[int(x) for x in np.linspace(start = 5, 
                                                                stop = 200, num = 5)],
                    "weights": ['uniform', 'distance'],
                    "algorithm": ['auto','ball_tree','kd_tree','brute'],
                    "leaf_size": np.arange(5,50,1)
                    }


"""
Estas listas no imcluyen la MLP porque se ejecuta a parte
"""
model_list = [ridge_reg,lasso_reg,elastic_net,tree_reg, random_forest,extra_tree,
               knreg]

param_grids = [param_grid_ridge,param_grid_lasso,param_grid_elastic,
               param_grid_decisiontree,param_grid_randomforest,
               param_grid_extratree, param_grid_knreg]





"""
Preparación de los datos para s2s
"""

data = pd.concat([imu_s2s_dataframe, emg_s2s_dataframe], axis=1)
data = data.reindex(sorted(data.columns), axis=1)
data_columns = data.columns

imu_s2s_ordered = imu_s2s_dataframe.reindex(sorted(imu_s2s_dataframe), axis = 1)
emg_s2s_ordered = emg_s2s_dataframe.reindex(sorted(emg_s2s_dataframe), axis = 1)

imu_ordered = imu_s2s_ordered.columns
emg_ordered = emg_s2s_ordered.columns

imu_l = imu_s2s_ordered[imu_ordered[0:72]]
imu_r = imu_s2s_ordered[imu_ordered[72:144]]
waist = imu_s2s_ordered[imu_ordered[144:]]

emg_r = emg_s2s_dataframe[[f for f in emg_s2s_dataframe.columns if f.startswith('R')]]
emg_l = emg_s2s_dataframe[[f for f in emg_s2s_dataframe.columns if f.startswith('L')]]

#emg_l = emg_s2s_ordered[emg_ordered[0:91]]
#emg_r = emg_s2s_ordered[emg_ordered[91:]]

imu_r = imu_r.drop_prefix('Right_')
imu_l = imu_l.drop_prefix('Left_')

emg_l = emg_l.drop_prefix('Left_')
emg_r = emg_r.drop_prefix('Right_')

waist_double = pd.concat([waist,waist], axis = 0)
imu_s2s = pd.concat([imu_r, imu_l], axis = 0)
imu_s2s = pd.concat([waist_double, imu_s2s], axis = 1)

emg_s2s = pd.concat([emg_r, emg_l], axis = 0)
emg_names = emg_s2s.columns

imu_s2s_ordered = imu_s2s.reindex(sorted(imu_s2s), axis = 1)
name_imu_ordered = imu_s2s_ordered.columns
emg_s2s_ordered = emg_s2s.reindex(sorted(emg_s2s), axis = 1)

"""
imu_elements = ["Right_Thigh","Left_Thigh",
                "Right_Shank","Left_Shank",
                'Waist']

emg_elements = ["Right_TA", "Left_TA",
                       "Right_MG", "Left_MG", "Right_SOL", "Left_SOL", "Right_BF", 
                       "Left_BF", "Right_ST", "Left_ST", "Right_VL", "Left_VL",
                       "Right_RF", "Left_RF"]
"""

imu_elements = ["Waist"]

emg_elements = ["TA","MG", "SOL", "BF", 
                       "ST", "VL",
                       "RF"]
possible_combinations = mcf.powerset(emg_elements)

desired_list = [("Waist",)+v for v in possible_combinations]

"""
Selección de las que tienen 5 sensores
"""
desired_list = [x for x in desired_list if len(x)==3]

imu_elements.sort()
emg_elements.sort()

"""
Separación de las variables por sensor para posibilitar probar todas las 
combinaciones
"""

imu_frag = {'Waist': imu_s2s_ordered[name_imu_ordered[72:]]}

emg_frag = mcf.create_subdataframes(emg_s2s_ordered, emg_elements,13)

y = emg_s2s.iloc[:, 21:28]

"""
Aplico todos los modelos a un conjunto pequeño de combinaciones para obtener un
modelo que introducir en la aplicación web y comprobar que todo se ejecuta 
correctamente
"""
dataset = desired_list
model_impl = mcf.bucle_completo(dataset, imu_frag, emg_frag, y, model_list, 
                                param_grids,param_grid_mlp)

"""
A continuación generamos la evaluación de los modelos anteriores
"""
models = ['Ridge', 'Lasso', 'ElasticNet', 'decisionTree', 'randomForest', 
          'extraTree', 'kneighbors','MLP']
metricas = ['rmse', 'mae', 'scores_cv', 'score.mean', 'scores.std:']

rmse = mcf.error_dataframe(model_impl['Error_measures'], dataset, models, 
                           metricas[0])    
mae = mcf.error_dataframe(model_impl['Error_measures'], dataset, models, 
                          metricas[1]) 
scores_mean = mcf.error_dataframe(model_impl['Error_measures'], dataset, 
                                  models, metricas[3]) 

       
df_rmse = pd.DataFrame(rmse[0], columns = models,  index = rmse[1])    
df_mae = pd.DataFrame(mae[0], columns = models,  index = mae[1])    
df_scores_mean = pd.DataFrame(scores_mean[0], columns = models,  
                              index = scores_mean[1])

"""
Generación del mejor modelo obtenido de la segunda combinación
"""
model_list_str =[str(i) for i in model_list]
para = mcf.obtencion_params(model_impl['Datos_modelos'], dataset[1], 
                            model_list_str[6]) 
best_model = mcf.obtencion_bestest(model_impl['Datos_modelos'], dataset[1], 
                                   model_list_str[6])

"""
Generación del archivo que almacena y permite subir el modelo a la aplicación web
"""
import pickle
pickle_out = open("multioutput_regression.pkl", mode = 'wb')

pickle.dump(best_model, pickle_out)
pickle_out.close()
