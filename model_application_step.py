# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:56:31 2021

@author: cjove
"""
import pandas as pd
import numpy as np
from numpy.random import randint, uniform
from itertools import chain, combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
#from implementation import dataframe_emg_step, dataframe_imu_step
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
Preparación de los datos para marcha en superficie plana, incluida la 
eliminación de NaN
"""


def drop_prefix(self, prefix):
    self.columns = self.columns.str.replace(prefix,'')
    return self

pd.core.frame.DataFrame.drop_prefix = drop_prefix

emg_step_right = dataframe_emg_step[[f for f in dataframe_emg_step.columns if f.startswith('R')]]
emg_step_left = dataframe_emg_step[[f for f in dataframe_emg_step.columns if f.startswith('L')]]

r = emg_step_right.drop_prefix('Right_')
l = emg_step_left.drop_prefix('Left_')

steps = pd.concat([r,l], axis=0)
steps_cols = steps.columns
steps.isnull().sum()
steps = steps.dropna()
steps_emg_ordered= steps.reindex(sorted(steps), axis = 1)

imu_cols = dataframe_imu_step.columns

r1 = dataframe_imu_step[dataframe_imu_step.columns[0:24]]
r2 = dataframe_imu_step[dataframe_imu_step.columns[48:60]]
l1 = dataframe_imu_step[dataframe_imu_step.columns[72:96]]
l2 = dataframe_imu_step[dataframe_imu_step.columns[144:156]]

r_imu = pd.concat([r1,r2], axis = 1)
l_imu = pd.concat([l1,l2], axis = 1)

r_imu = r_imu.drop_prefix('Right_')
l_imu = l_imu.drop_prefix('Left_')

steps_imu = pd.concat([r_imu, l_imu], axis = 0)
steps_imu.isnull().sum()
steps_imu = steps_imu.dropna()
steps_imu_ordered = steps_imu.reindex(sorted(steps_imu), axis = 1)

steps_imu.isnull().sum()


imu_elements_step = ["Shank"]

emg_elements_step = ["TA","MG", "SOL", "BF", 
                       "ST", "VL",
                       "RF"]

possible_combinations_step = mcf.powerset(emg_elements_step)

desired_list_step = [("Shank",)+v for v in possible_combinations_step]

"""
Selección de las que tienen 3 sensores
"""

dataset_step = [x for x in desired_list_step if len(x)==3]

"""
Separación de las variables por sensor para posibilitar probar todas las 
combinaciones
"""
imu_elements_step.sort()
emg_elements_step.sort()


imu_frag_step = mcf.create_subdataframes(steps_imu_ordered, imu_elements_step,36)
emg_frag_step = mcf.create_subdataframes(steps_emg_ordered, emg_elements_step,13)

y_step = steps.iloc[:,21:28]

"""
Aplico todos los modelos a un conjunto pequeño de combinaciones para obtener un
modelo que introducir en la aplicación web y comprobar que todo se ejecuta 
correctamente
"""


model_impl_step = mcf.bucle_completo(dataset_step, imu_frag_step, emg_frag_step, y_step, model_list, param_grids,param_grid_mlp)

"""
A continuación generamos la evaluación de los modelos anteriores
"""
models = ['Ridge', 'Lasso', 'ElasticNet', 'decisionTree', 'randomForest', 
          'extraTree', 'kneighbors','MLP']
metricas = ['rmse', 'mae', 'scores_cv', 'score.mean', 'scores.std:']

rmse_step = mcf.error_dataframe(model_impl_step['Error_measures'], dataset_step, models, metricas[0])    
mae_step = mcf.error_dataframe(model_impl_step['Error_measures'], dataset_step, models, metricas[1]) 
scores_mean_step = mcf.error_dataframe(model_impl_step['Error_measures'], dataset_step, models, metricas[3]) 



df_rmse_step = pd.DataFrame(rmse_step[0], columns = models,  index = rmse_step[1])    
df_mae_step = pd.DataFrame(mae_step[0], columns = models,  index = mae_step[1])    
df_scores_mean_step = pd.DataFrame(scores_mean_step[0], columns = models,  index = scores_mean_step[1])

"""
Generación del mejor modelo obtenido de la segunda combinación
"""

model_list_str =[str(i) for i in model_list]

best_model_step = mcf.obtencion_bestest(model_impl_step['Datos_modelos'], dataset_step[1], model_list_str[6])

"""
Generación del archivo que almacena y permite subir el modelo a la aplicación web
"""

import pickle
pickle_out = open("multioutput_regression_step.pkl", mode = 'wb')
    
pickle.dump(best_model_step, pickle_out)
pickle_out.close()






