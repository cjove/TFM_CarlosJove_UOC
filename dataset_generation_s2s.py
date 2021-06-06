# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:50:02 2021

@author: cjove
"""
from sit2stand_functions import sit_to_stand_threshold, peaks_gyro, sit_to_stand
import variable_extraction_s2s as ve_s2s
import numpy as np
import pandas as pd

def acc_features_sit2stand(data,acc_names):
    print('Empezando la generación del dataset del sensor inercial por levantamiento')

    thres_points = sit_to_stand_threshold(data)
    peaks = peaks_gyro(data)
    s2s = sit_to_stand(peaks, thres_points)
    
    general = []
    ini_fin = []
    
    for x in acc_names:
        gen = ve_s2s.min_max_mean_std(data[x], s2s)
        inifin = ve_s2s.initial_final(data[x], s2s)
        general.append(gen)
        ini_fin.append(inifin)
    print('Finalizada la generación del dataset del sensor inercial por levantamiento')
    
    return(general, ini_fin)

def emg_features_sit2stand(data, muscle_names, data_gy):
    print('Empezando la generación del dataset de electromiografía por levantamiento')

    thres_points = sit_to_stand_threshold(data_gy)
    peaks = peaks_gyro(data_gy)
    s2s = sit_to_stand(peaks, thres_points)
    
    mav = []
    ssi = []
    v = []
    rms = []
    wfl = []
    zc = []
    ssc = []
    wa = []
    mdf = []
    mf = []
    she = []
    spe = []
    svde = []
    
    for x in muscle_names:
        mav_ = ve_s2s.mean_absolute_value(data[x], s2s)
        ssi_ = ve_s2s.simple_square_integral(data[x], s2s)
        v_ = ve_s2s.variance(data[x],s2s)
        rms_ = ve_s2s.root_mean_square(data[x],s2s)
        wfl_ = ve_s2s.waveform_length(data[x],s2s)
        zc_ = ve_s2s.zero_crossing(data[x],s2s)
        ssc_ = ve_s2s.slope_sign_changes(data[x],s2s)
        wa_ = ve_s2s.willison_amplitude(data[x],s2s)
        mdf_ = ve_s2s.median_frequency(data[x],s2s)
        mf_ = ve_s2s.mean_frequency(data[x],s2s)
        she_ = ve_s2s.shannon_entropy(data[x],s2s)
        spe_ = ve_s2s.spectral_entropy(data[x],s2s)
        svde_ = ve_s2s.singlevaluedecomp_entropy(data[x],s2s)
    
        mav.append(mav_)
        ssi.append(ssi_)
        v.append(v_)
        rms.append(rms_)
        wfl.append(wfl_)
        zc.append(zc_)
        ssc.append(ssc_)
        wa.append(wa_)
        mdf.append(mdf_)
        mf.append(mf_)
        she.append(she_)
        spe.append(spe_)
        svde.append(svde_)
    print('Finalizada la generación del dataset de electromiografía por levantamiento')
    
    return(mav,
    ssi,
    v,
    rms,
    wfl,
    zc ,
    ssc,
    wa ,
    mdf,
    mf ,
    she,
    spe,
    svde)

def create_dataframe_s2s(data,labels,info):
    dicc = {}
    if info == 0:
        print('Generando la matriz de datos del sensor inercial para transición de sedestación a bipedestación')

        listado = np.arange(0,len(data),1)
        for x in listado:
            if x == 0:
                length = np.arange(0,len(data[x]),1) 
                pos = 0
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[labels[num]]= data[x][b][c]
                    pos = pos + 6
            elif x == 1:

                length = np.arange(0,len(data[x]),1) 
                pos = 4
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[labels[num]]= data[x][b][c]
                    pos = pos + 6
    else:
        print('Generando la matriz de datos de electromiografía para para transición de sedestación a bipedestación')        
        listado = np.arange(0, len(data),1)
        pos = 0
        for x in listado:
            length = np.arange(0,len(data[x]),1)
            for b in length:
                num = int(b + pos)
                dicc[labels[num]] = data[x][b]
            pos = pos + 14
            
    dataframe = pd.DataFrame.from_dict(dicc)
    print('Matriz generada con éxito')        
        
    return dataframe