# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:36:11 2021

@author: cjove
"""

import model_creation_functions as mcf

"""Implementación del modelo con selección de variables"""

model_impl_step_rfe = mcf.bucle_completo_rfe(dataset_step, imu_frag_step, emg_frag_step, y_step, model_list, param_grids,param_grid_mlp)

"""s2s rffe"""

model_impl_s2s_rfe = mfc.bucle_completo_rfe(dataset, imu_frag, emg_frag, y, model_list, param_grids,param_grid_mlp)


"""Implementación del modelo sin el resto de datos del acelerometro"""

"""Step"""
"""
param_grid_mlp_acc = {
 "n_hidden":[0,1,2,3,4],
 "n_neurons": np.arange(1,100).tolist(),
 "learning_rate": np.arange(3e-4,3e-2,0.005).tolist(),
 "input_shape": (32,)
 }
"""
imu_frag_step_gy = imu_frag_step['Shank'][[f for f in imu_frag_step['Shank'].columns if f.startswith('Shank_Gy')]]

imu_frag_step_gy_dict = {}
imu_frag_step_gy_dict['Shank']= imu_frag_step_gy
model_impl_step_gy = mcf.bucle_completo(dataset_step, imu_frag_step_gy_dict, emg_frag_step, y_step, model_list, param_grids,param_grid_mlp)

"""S2S"""
model_impl_s2s_rfe = mcf.bucle_completo_rfe(dataset, imu_frag, emg_frag, y, model_list, param_grids,param_grid_mlp)
imu_frag_gx = imu_frag['Waist'][[f for f in imu_frag['Waist'].columns if f.startswith('Waist_Gx')]]

imu_frag_gx_dict = {}
imu_frag_gx_dict['Waist']= imu_frag_gx
model_impl_s2s_gx = mcf.bucle_completo(dataset, imu_frag_gx_dict, emg_frag, y, model_list, param_grids,param_grid_mlp)


"""Implementación del modelo sobre los datos sin outliers"""


imu_frag_step_out = mcf.create_subdataframes(out_iso_imu_step, imu_elements_step,36)
emg_frag_step_out = mcf.create_subdataframes(out_iso_emg_step, emg_elements_step,13)



model_impl_step_out = mcf.bucle_completo(dataset_step, imu_frag_step_out, emg_frag_step_out, y_step_out, model_list, param_grids,param_grid_mlp)




imu_frag_s2s_out = {'Waist': out_iso_imu_s2s[names_imu_s2s[72:]]}
emg_frag_s2s_out = mcf.create_subdataframes(out_iso_emg_s2s, emg_elements,13)

model_impl_s2s_out = mcf.bucle_completo(dataset, imu_frag_s2s_out, emg_frag_s2s_out, y_out, model_list, 
                                param_grids,param_grid_mlp)











"""Función para los resultados"""

def result_dataframes (modelo, dataset, models, metricas):

    rmse = mcf.error_dataframe(modelo['Error_measures'], dataset, models, 
                               metricas[0])    
    mae = mcf.error_dataframe(modelo['Error_measures'], dataset, models, 
                              metricas[1]) 
    scores_mean = mcf.error_dataframe(modelo['Error_measures'], dataset, 
                                      models, metricas[3]) 
    
           
    df_rmse = pd.DataFrame(rmse[0], columns = models,  index = rmse[1])    
    df_mae = pd.DataFrame(mae[0], columns = models,  index = mae[1])    
    df_scores_mean = pd.DataFrame(scores_mean[0], columns = models,  
                                  index = scores_mean[1])
    
    return {'RMSE' :df_rmse, 'MAE' : df_mae, "Mean scores": df_scores_mean}

models = ['Ridge', 'Lasso', 'ElasticNet', 'decisionTree', 'randomForest', 
          'extraTree', 'kneighbors','MLP']
metricas = ['rmse', 'mae', 'scores_cv', 'score.mean', 'scores.std:']


resultados_step = result_dataframes(model_impl_step,dataset_step, models, metricas)
resultados_s2s = result_dataframes(model_impl,dataset, models, metricas)
resultados_step_rfe = result_dataframes(model_impl_step_rfe,dataset_step, models, metricas)
resultados_s2s_rfe = result_dataframes(model_impl_s2s_rfe,dataset, models, metricas)
resultados_step_out = result_dataframes(model_impl_step_out,dataset_step, models, metricas)
resultados_s2s_out = result_dataframes(model_impl_s2s_out,dataset, models, metricas)
resultados_step_gy = result_dataframes(model_impl_step_gy,dataset_step, models, metricas)
resultados_s2s_gx = result_dataframes(model_impl_s2s_gx,dataset, models, metricas)

"""
Obtención de las variables que han sido seleccionadas por selección recursiva de variables
"""
    
s2s_rfe_selfeat = model_impl_s2s_rfe['Selected features']
step_rfe_selfeat = model_impl_step_rfe['Selected features']

"""
Generación de los modelos seleccionados tras la evaluación de los datos obtenidos
"""

model_list_str =[str(i) for i in model_list]
para_step = mcf.obtencion_params(model_impl_step_gy['Datos_modelos'], dataset_step[4], 
                            model_list_str[5]) 
best_model_step = mcf.obtencion_bestest(model_impl_step_gy['Datos_modelos'], dataset_step[4], 
                                   model_list_str[5])


para_s2s = mcf.obtencion_params(model_impl_s2s_rfe['Datos_modelos'], dataset[2], 
                            model_list_str[5]) 
best_model_s2s = mcf.obtencion_bestest(model_impl_s2s_rfe['Datos_modelos'], dataset[2], 
                                   model_list_str[5])


"""
Generación del archivo que almacena y permite subir el modelo a la aplicación web
"""
import pickle
pickle_out = open("multioutput_regression_step_def.pkl", mode = 'wb')

pickle.dump(best_model_step, pickle_out)
pickle_out.close()

pickle_out_2 = open("multioutput_regression_s2s_def.pkl", mode = 'wb')

pickle.dump(best_model_s2s, pickle_out_2)
pickle_out.close()