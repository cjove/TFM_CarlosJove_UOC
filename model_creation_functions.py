# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:51:10 2021

@author: cjove
"""
from itertools import chain, combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.stats import uniform as sp_rand
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFECV
import tensorflow as tf

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def create_subdataframes(data, labels, salto):
    space = np.arange(0,len(data.columns)+1,salto)
    len_labels = np.arange(0,len(labels),1)
    dataframes = []
    dict_data = {}
    for i in space:
        sub = data.iloc[:,i:i+salto]
        dataframes.append(sub)
    for i in len_labels:
       dict_data[labels[i]]=dataframes[i]


    return dict_data

def combination(data_labels, imu_frag, emg_frag, y):
    length_2 = np.arange(0,len(data_labels),1)
    #print(length_2)
    x = []
    labels = []
    for ii in length_2:
        if data_labels[ii] in emg_frag:
            x.append(emg_frag[data_labels[ii]])
            label = data_labels[ii]+'_rms'
            labels.append(label)


        else:
            x.append(imu_frag[data_labels[ii]])
            
    y = y.drop(columns = labels)            
    X = pd.concat(x, axis= 1)

    return X,y

def data_preparation(X,y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    scaler.fit(test_X)
    test_X = scaler.transform(test_X)
    

    #return {'train_X': train_X, 'test_X': test_X, 'train_y': train_y, 'test_y': test_y}
    return train_X, test_X,  train_y, test_y

def aplicar_modelos(model_list, param_grids,X,y,max_features):
    results = []
    length = np.arange(0,len(model_list),1)
    for x in length:
        print('Entrenando el modelo '+ str(model_list[x]))
        res = RandomizedSearchCV(model_list[x],param_grids[x], cv = 30,
                                      scoring="neg_mean_squared_error",
                                      return_train_score = True,
                                      n_jobs=-1
                                      )
        res.fit(X, y)
        ht_params = res.best_params_
        ht_score = res.best_score_
        best_est = res.best_estimator_
        print('Modelo' + str(model_list[x]) +'entrenado!')
        
        
        results.append({'best_params': ht_params, 'ht_score': ht_score, 'best_estimator': best_est})
    model_list_str =[str(i) for i in model_list]
    results = dict(zip(model_list_str, results))    
    
    return results 

def aplicar_mlp(param_grids, X, y, max_features, cols_y):
    #modelo = build_mlp(input_shape = max_features, output_shape = cols_y)
    
    def build_mlp( n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape = max_features):
        #Validate if the input_shape is correct
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape = input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(5))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    #mlp_reg = keras.wrappers.scikit_learn.KerasRegressor(modelo)
    mlp_reg = keras.wrappers.scikit_learn.KerasRegressor(build_mlp)
    res = RandomizedSearchCV(mlp_reg,param_grids, cv = 3,
                             n_iter= 10,
                             n_jobs=-1
                             )
    #X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=42)
    
    #res.fit(X_train, y_train,  epochs = 100, validation_data = (X_valid, y_valid), callbacks = [keras.callbacks.EarlyStopping(monitor='loss',patience=10)])
    res.fit(X, y,  epochs = 100, callbacks = [keras.callbacks.EarlyStopping(monitor='loss',patience=10)])
    ht_params = res.best_params_
    ht_score = res.best_score_
    best_est = res.best_estimator_
    #Esto es el último cambio
    #best_est = res.best_estimator_


    return {'best_params_mlp': ht_params, 'ht_score_mlp': ht_score, 'best_estimator': best_est}




def bucle_completo(desired_list, imu_frag, emg_frag, y,model_list,param_grids, param_grid_mlp):
    datos = {}
    datos_mlp = {}
    error_measure = {}    
    length = np.arange(0,len(desired_list),1)
    for i in length:
        labels = desired_list[i]
        print('Empezando con el grupo:' +str(labels))
        X,y1 = combination(labels,imu_frag, emg_frag,y)
        cols = X.columns
        cols_2 = y1.columns
        
        max_features =X.shape[1:]
        cols_y = len(y1.columns)
        print(max_features)
        train_X, test_X,  train_y, test_y = data_preparation(X,y1)
        resultados = aplicar_modelos(model_list,param_grids, train_X,train_y, max_features)
        print('Empezando a entrenar la mlp en el grupo: '+str(labels))
        resultados_mlp = aplicar_mlp(param_grid_mlp, train_X, train_y, max_features, cols_y)
        print('MLP entrenada!')
        predictions = prediction_models(resultados,resultados_mlp,train_X, train_y, test_X, test_y,X,y1, max_features)
        datos[labels] = resultados
        datos_mlp[labels] = resultados_mlp
        error_measure[labels] = predictions

    return {'Datos_modelos':datos,'Datos_MLP':datos_mlp,'Error_measures': error_measure}

def prediction_models (resultados,resultados_mlp, train_X, train_y, test_X, test_y, X, y1, max_features):
    #models = resultados[0]
    #mlp = resultados[1]
    #length = np.arange(0,len(desired_list),1)

    #for i in length:
    #   labels = desired_list[i]
    #values_models = models[desired_list[i]]
    ridge = Ridge(alpha = resultados['Ridge()']['best_params']['alpha'])
    lasso = Lasso(alpha = resultados['Lasso()']['best_params']['alpha'])
    elastic = ElasticNet(alpha = resultados['ElasticNet(alpha=0.1)']['best_params']['alpha'],
                         l1_ratio = resultados['ElasticNet(alpha=0.1)']['best_params']['l1_ratio'])
    decisionTree = DecisionTreeRegressor(max_features = resultados['DecisionTreeRegressor()']['best_params']['max_features'],
                                         min_samples_split = resultados['DecisionTreeRegressor()']['best_params']['min_samples_split'],
                                         max_depth = resultados['DecisionTreeRegressor()']['best_params']['max_depth'],
                                         min_samples_leaf = resultados['DecisionTreeRegressor()']['best_params']['min_samples_leaf']
                                        )
    randomForest = RandomForestRegressor(n_estimators= resultados['RandomForestRegressor()']['best_params']['n_estimators'],
                                         max_features = resultados['RandomForestRegressor()']['best_params']['max_features'],
                                         max_depth = resultados['RandomForestRegressor()']['best_params']['max_depth'] ,
                                         min_samples_split = resultados['RandomForestRegressor()']['best_params']['min_samples_split'],
                                         min_samples_leaf = resultados['RandomForestRegressor()']['best_params']['min_samples_leaf'],
                                         bootstrap =resultados['RandomForestRegressor()']['best_params']['bootstrap'])
    extraTree = ExtraTreesRegressor(n_estimators = resultados['ExtraTreesRegressor()']['best_params']['n_estimators'],
                                    criterion = resultados['ExtraTreesRegressor()']['best_params']['criterion'] ,
                                    max_depth = resultados['ExtraTreesRegressor()']['best_params']['max_depth'],
                                    min_samples_split = resultados['ExtraTreesRegressor()']['best_params']['min_samples_split'],
                                    min_samples_leaf = resultados['ExtraTreesRegressor()']['best_params']['min_samples_leaf'],
                                    bootstrap = resultados['ExtraTreesRegressor()']['best_params']['bootstrap'])
    kneighbors = KNeighborsRegressor(n_neighbors = resultados['KNeighborsRegressor()']['best_params']['n_neighbors'],
                                     weights = resultados['KNeighborsRegressor()']['best_params']['weights'],
                                     algorithm = resultados['KNeighborsRegressor()']['best_params']['algorithm'],
                                     leaf_size =resultados['KNeighborsRegressor()']['best_params']['leaf_size'])
    

    def build_mlp( n_hidden=resultados_mlp['best_params_mlp']['n_hidden'], n_neurons=resultados_mlp['best_params_mlp']['n_neurons'], learning_rate=resultados_mlp['best_params_mlp']['learning_rate'], input_shape = max_features):
        #Validate if the input_shape is correct
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape = input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(5))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    #mlp_reg = keras.wrappers.scikit_learn.KerasRegressor(modelo)
    model = keras.wrappers.scikit_learn.KerasRegressor(build_mlp)
    
    print('Ridge cross-val')
    ridge_scores = cross_val_score(ridge, X, y1, scoring = "neg_mean_squared_error", cv = 10, n_jobs=-1)
    ridge_rmse_scores = np.sqrt(-ridge_scores)
    print('Lasso cross-val')
    lasso_scores = cross_val_score(lasso, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    lasso_rmse_scores = np.sqrt(-lasso_scores)
    print('Elastic cross-val')
    elastic_scores = cross_val_score(elastic, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    elastic_rmse_scores = np.sqrt(-elastic_scores)
    print('DecisionTree cross-val')
    decisionTree_scores = cross_val_score(decisionTree, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    decisionTree_rmse_scores = np.sqrt(-decisionTree_scores)
    print('RandomForest cross-val')
    randomForest_scores = cross_val_score(randomForest, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    randomForest_rmse_scores = np.sqrt(-randomForest_scores)   
    print('ExtraTree cross-val')     
    extraTree_scores = cross_val_score(extraTree, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    extraTree_rmse_scores = np.sqrt(-extraTree_scores) 
    print('kneighbours cross-val')
    kneighbors_scores = cross_val_score(kneighbors, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    kneighbors_rmse_scores = np.sqrt(-kneighbors_scores) 
    
    print('MLP cross-val')
    model = resultados_mlp['best_estimator']
    mlp_scores = cross_val_score(model, X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    mlp_scores = cross_val_score(resultados_mlp['best_estimator'], X, y1, scoring = "neg_mean_squared_error", cv = 10,n_jobs=-1)
    mlp_rmse_scores = np.sqrt(-mlp_scores) 
    print('cross-val ended')

    
    print('Ridge fit')
    ridge.fit(train_X,train_y)
    print('Lasso fit')
    lasso.fit(train_X,train_y)
    print('Elastic fit')
    elastic.fit(train_X,train_y)
    print('DecisionTree fit')
    decisionTree.fit(train_X,train_y)
    print('RandomForest fit')
    randomForest.fit(train_X,train_y)
    print('Extratree fit')
    extraTree.fit(train_X,train_y)
    print('Kneighbours fit')
    kneighbors.fit(train_X,train_y)
    print('Fit ended')
    
    
    
    print('Ridge predict')
    ridge_predict = ridge.predict(test_X)
    print('Lasso predict')
    lasso_predict = lasso.predict(test_X)
    print('Elastic predict')
    elastic_predict = elastic.predict(test_X)
    print('DecisionTree predict')
    decisionTree_predict = decisionTree.predict(test_X)
    print('RandomForest predict')
    randomForest_predict = randomForest.predict(test_X)
    print('ExtraTree predict')
    extraTree_predict = extraTree.predict(test_X)
    print('Kneighbours predict')
    kneighbors_predict = kneighbors.predict(test_X)
    print('Predict ended')
    
    print('Ridge errors')
    ridge_mse = mean_squared_error(test_y, ridge_predict)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_mae = mean_absolute_error(test_y, ridge_predict)
    print('Lasso errors')
    lasso_mse = mean_squared_error(test_y, lasso_predict)
    lasso_rmse = np.sqrt(lasso_mse)
    lasso_mae = mean_absolute_error(test_y, lasso_predict)
    print('Elastic errors')
    elastic_mse = mean_squared_error(test_y, elastic_predict)
    elastic_rmse = np.sqrt(elastic_mse)
    elastic_mae = mean_absolute_error(test_y, elastic_predict)
    print('DecisionTrees Errors')
    decisionTree_mse = mean_squared_error(test_y, decisionTree_predict)
    decisionTree_rmse = np.sqrt(decisionTree_mse)
    decisionTree_mae = mean_absolute_error(test_y, decisionTree_predict)
    print('RandomForest Errors')
    randomForest_mse = mean_squared_error(test_y, randomForest_predict)
    randomForest_rmse = np.sqrt(randomForest_mse)
    randomForest_mae = mean_absolute_error(test_y, randomForest_predict)
    print('ExtraTree Errors')
    extraTree_mse = mean_squared_error(test_y, extraTree_predict)
    extraTree_rmse = np.sqrt(extraTree_mse)
    extraTree_mae = mean_absolute_error(test_y, extraTree_predict)
    print('Kneighbours Errors')
    kneighbors_mse = mean_squared_error(test_y, kneighbors_predict)
    kneighbors_rmse = np.sqrt(kneighbors_mse)
    kneighbors_mae = mean_absolute_error(test_y, kneighbors_predict)
    

    print('MLP fit')
    model.fit(train_X,train_y)
    print('MLP predict')
    mlp_predict = model.predict(test_X)
    print('MLP errors')
    mlp_mse = mean_squared_error(test_y, mlp_predict)
    mlp_rmse = np.sqrt(mlp_mse)
    mlp_mae = mean_absolute_error(test_y, mlp_predict)
    print('Prediction models ended')
    
    
    
    return {'Ridge':{'rmse':ridge_rmse,'mae':ridge_mae,'scores_cv':ridge_rmse_scores ,'score.mean': ridge_rmse_scores.mean(),'scores.std:':ridge_rmse_scores.std()},
                    'Lasso':{'rmse':lasso_rmse,'mae':lasso_mae,'scores_cv':lasso_rmse_scores ,'score.mean': lasso_rmse_scores.mean(),'scores.std:':lasso_rmse_scores.std()},
                    'ElasticNet':{'rmse':elastic_rmse,'mae':elastic_mae,'scores_cv':elastic_rmse_scores ,'score.mean': elastic_rmse_scores.mean(),'scores.std:':elastic_rmse_scores.std()},
                    'decisionTree':{'rmse':decisionTree_rmse,'mae':decisionTree_mae,'scores_cv':decisionTree_rmse_scores ,'score.mean': decisionTree_rmse_scores.mean(),'scores.std:':decisionTree_rmse_scores.std()},
                    'randomForest':{'rmse':randomForest_rmse,'mae':randomForest_mae,'scores_cv':randomForest_rmse_scores ,'score.mean': randomForest_rmse_scores.mean(),'scores.std:':randomForest_rmse_scores.std()},
                    'extraTree':{'rmse':extraTree_rmse,'mae':extraTree_mae,'scores_cv':extraTree_rmse_scores ,'score.mean': extraTree_rmse_scores.mean(),'scores.std:':extraTree_rmse_scores.std()},
                    'kneighbors':{'rmse':kneighbors_rmse,'mae':kneighbors_mae,'scores_cv':kneighbors_rmse_scores ,'score.mean': kneighbors_rmse_scores.mean(),'scores.std:':kneighbors_rmse_scores.std()},
                    'MLP':{'rmse':mlp_rmse,'mae':mlp_mae,'scores_cv':mlp_rmse_scores ,'score.mean': mlp_rmse_scores.mean(),'scores.std:':mlp_rmse_scores.std()}
                    }


def error_dataframe(error_measures, desired_list, models, metrica):
    length = np.arange(0,len(desired_list),1)
    data = []
    names = []
       
    for i in length:
        pack = desired_list[i]
        packs = str(pack)
        length_2 = np.arange(0,len(models),1)
        errors = []
        for i in length_2:
            #model = models[i]
            error = error_measures[pack][models[i]][metrica]
            #mae = error_measures[pack][models[i]]['mae']
            #scores_cv = error_measures[pack][models[i]]['scores_cv']
            #scores_mean = error_measures[pack][models[i]]['score.mean']
            #scores_std = error_measures[pack][models[i]]['scores.std:']
            #list_errors = [model,rmse, mae, scores_cv, scores_mean, scores_std]
            errors.append(error)
        names.append(packs)
        #errors.append(pack)
        data.append(errors)
        
    return data , names

def obtencion_params(datos,combinacion,modelo):
    params = datos[combinacion][modelo]['best_params']
    
    return params

def obtencion_bestest(datos,combinacion,modelo):
    params = datos[combinacion][modelo]['best_estimator']
    
    return params

def feature_selection(X, y):
    rfe =RFECV(estimator=DecisionTreeRegressor())
    rfe.fit(X, y)
    features = rfe.support_
    numeric = np.arange(0, len(features),1)
    numeric = list(numeric[features])



    return numeric



def bucle_completo_rfe(desired_list, imu_frag, emg_frag, y,model_list,param_grids, param_grid_mlp):
    datos = {}
    datos_mlp = {}
    error_measure = {}   
    features_selected = {}


    length = np.arange(0,len(desired_list),1)
    for i in length:
        labels = desired_list[i]
        print('Empezando con el grupo:' +str(labels))
        X,y1 = combination(labels,imu_frag, emg_frag,y)
        cols = X.columns
        cols_array = np.asarray(cols)
        #max_features = int(len(cols))
        cols_y = len(y1.columns)

        train_X, test_X,  train_y, test_y = data_preparation(X,y1)
        print(train_X.shape, train_y.shape)
        features = feature_selection(train_X, train_y)
        features_names = list(cols_array[features])
        train_X = train_X[:,np.asarray(features)]
        print(train_X.shape)
        test_X = test_X[:,np.asarray(features)]
        X = X[features_names]
        max_features = train_X.shape[1:]
        print(max_features)

        resultados = aplicar_modelos(model_list,param_grids, train_X,train_y, max_features)
        print('Empezando a entrenar la mlp en el grupo: '+str(labels))
        resultados_mlp = aplicar_mlp(param_grid_mlp, train_X, train_y, max_features, cols_y)
        print('MLP entrenada!')
        predictions = prediction_models(resultados,resultados_mlp,train_X, train_y, test_X, test_y,X,y1, max_features)
        datos[labels] = resultados
        datos_mlp[labels] = resultados_mlp
        error_measure[labels] = predictions
        features_selected[labels] = features_names

    return {'Datos_modelos':datos,'Datos_MLP':datos_mlp,'Error_measures': error_measure, 'Selected features': features_selected}


def aplicar_mlp_rfe(param_grids, X, y, max_features, cols_y):
    def build_mlp( n_hidden=1, n_neurons=30, learning_rate=3e-3):
        #Validate if the input_shape is correct
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape = max_features))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(5))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    mlp_reg = keras.wrappers.scikit_learn.KerasRegressor(build_mlp)
    res = RandomizedSearchCV(mlp_reg,param_grids, cv = 3,
                             n_iter= 10,
                             n_jobs=-1
                             )
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=42)
    
    res.fit(X_train, y_train,  epochs = 100, validation_data = (X_valid, y_valid), callbacks = [keras.callbacks.EarlyStopping(monitor='loss',patience=10)])
    ht_params = res.best_params_
    ht_score = res.best_score_
    best_est = res.best_estimator_


    return {'best_params_mlp': ht_params, 'ht_score_mlp': ht_score, 'best_estimator': best_est}


