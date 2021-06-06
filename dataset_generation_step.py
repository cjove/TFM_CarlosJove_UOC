# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:21:51 2021

@author: cjove
"""

"""
Created on Tue Apr 13 18:42:36 2021
@author: cjove
"""

"""
Arbitrariamente he decidido incluir con los pasos de la izquierda los valores
del IMU de la cadera
"""
import pandas as pd
import numpy as np
from step_functions import invertir, gait_cycles
import variable_extraction_step as ve

def acc_features_step(data, acc_names, index_steps):
    print('Empezando la generación del dataset del sensor inercial por paso')

    gy_invertidos_d = invertir(data['Right_Shank_Gy'], index_steps)
    gy_invertidos_i = invertir(data['Left_Shank_Gy'], index_steps)
    gait_d = gait_cycles(gy_invertidos_d, index_steps)
    gait_i = gait_cycles(gy_invertidos_i, index_steps)

    acc_names_d = acc_names[0:12]
    acc_names_i = acc_names[12:]
    
    generald = []
    generali = []
    ini_find = []
    ini_fini = []
    
    for x in acc_names_d:
        general_d = ve.min_max_mean_std(data[x], gait_d)
        ini_fin_d = ve.initial_final(data[x], gait_d)
        generald.append(general_d)
        ini_find.append(ini_fin_d)
        
    for x in acc_names_i:
        general_i = ve.min_max_mean_std(data[x], gait_i)
        ini_fin_i = ve.initial_final(data[x], gait_i)
        generali.append(general_i)
        ini_fini.append( ini_fin_i)
    
    print('Finalizada la generación del dataset del sensor inercial por paso')
    
    return(generald,  ini_find,generali, ini_fini)


def emg_features_step(data, muscle_names, index_steps,  data_gy_d ,data_gy_i):
    print('Empezando la generación del dataset de electromiografía por paso')

    gy_invertidos_d = invertir(data_gy_d, index_steps)
    gy_invertidos_i = invertir(data_gy_i, index_steps)
    gait_d = gait_cycles(gy_invertidos_d, index_steps)
    gait_i = gait_cycles(gy_invertidos_i, index_steps)
    
    mavd = []
    ssid = []
    vd = []
    rmsd = []
    wfld = []
    zcd = []
    sscd = []
    wad = []
    mdfd = []
    mfd = []
    shed = []
    sped = []
    svded = []
    
    mavi = []
    ssii = []
    vi = []
    rmsi = []
    wfli = []
    zci = []
    ssci = []
    wai = []
    mdfi = []
    mfi = []
    shei = []
    spei = []
    svdei = []
    
    muscle_names_d = muscle_names[0:7]
    muscle_names_i = muscle_names[7:]
    
    for x in muscle_names_d:
        mav_d = ve.mean_absolute_value(data[x], gait_d)
        ssi_d = ve.simple_square_integral(data[x], gait_d)
        v_d = ve.variance(data[x],gait_d)
        rms_d = ve.root_mean_square(data[x],gait_d)
        wfl_d = ve.waveform_length(data[x],gait_d)
        zc_d = ve.zero_crossing(data[x],gait_d)
        ssc_d = ve.slope_sign_changes(data[x],gait_d)
        wa_d = ve.willison_amplitude(data[x],gait_d)
        mdf_d = ve.median_frequency(data[x],gait_d)
        mf_d = ve.mean_frequency(data[x],gait_d)
        she_d = ve.shannon_entropy(data[x],gait_d)
        spe_d = ve.spectral_entropy(data[x],gait_d)
        svde_d = ve.singlevaluedecomp_entropy(data[x],gait_d)
    
        mavd.append(mav_d)
        ssid.append(ssi_d)
        vd.append(v_d)
        rmsd.append(rms_d)
        wfld.append(wfl_d)
        zcd.append(zc_d)
        sscd.append(ssc_d)
        wad.append(wa_d)
        mdfd.append(mdf_d)
        mfd.append(mf_d)
        shed.append(she_d)
        sped.append(spe_d)
        svded.append(svde_d)
        
    for x in muscle_names_i:
        mav_i = ve.mean_absolute_value(data[x], gait_i)
        ssi_i = ve.simple_square_integral(data[x], gait_i)
        v_i = ve.variance(data[x],gait_i)
        rms_i = ve.root_mean_square(data[x],gait_i)
        wfl_i = ve.waveform_length(data[x],gait_i)
        zc_i = ve.zero_crossing(data[x],gait_i)
        ssc_i = ve.slope_sign_changes(data[x],gait_i)
        wa_i = ve.willison_amplitude(data[x],gait_i)
        mdf_i = ve.median_frequency(data[x],gait_i)
        mf_i = ve.mean_frequency(data[x],gait_i)
        she_i = ve.shannon_entropy(data[x],gait_i)
        spe_i = ve.spectral_entropy(data[x],gait_i)
        svde_i = ve.singlevaluedecomp_entropy(data[x],gait_i)
        
        mavi.append(mav_i)
        ssii.append(ssi_i)
        vi.append(v_i)
        rmsi.append(rms_i)
        wfli.append(wfl_i)
        zci.append(zc_i)
        ssci.append(ssc_i)
        wai.append(wa_i)
        mdfi.append(mdf_i)
        mfi.append(mf_i)
        shei.append(she_i)
        spei.append(spe_i)
        svdei.append(svde_i)

    print('Finalizada la generación del dataset de electromiografía por paso')
        
    return(mavd,
    ssid, 
    vd, 
    rmsd,
    wfld,
    zcd ,
    sscd,
    wad ,
    mdfd,
    mfd ,
    shed, 
    sped, 
    svded, 
    mavi, 
    ssii, 
    vi ,
    rmsi, 
    wfli, 
    zci ,
    ssci,
    wai ,
    mdfi,
    mfi,
    shei,
    spei, 
    svdei)

def create_labels_imu(names, variables):
    labels = []
    for x in names:
        for y in variables:
            labels.append(str(x+'_'+y))
    return labels

def create_labels_emg(names, variables):
    labels = []
    for x in variables:
        for y in names:
            labels.append(str(y+'_'+x))
    return labels


featurename_emg = ['mav','ssi','v', 'rms','wfl','zc' ,'ssc','wa' ,
               'mdf','mf' ,'she', 'spe', 'svde']

featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']


def create_dataframe_step(data, labels, info):
    #length = np.arange(0,len(data),1)

    dicc = {}
    if info == 0:
        print('Generando la matriz de datos del sensor inercial marcha en plano')

        labels_d = labels[0:72]
        labels_i = labels[72:]
        listado = np.arange(0,len(data),1)
        for x in listado:
            if x == 0:
                lab = labels_d
                length = np.arange(0,len(data[x]),1) 
                pos = 0
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[lab[num]]= data[x][b][c]
                    pos = pos + 6
            elif x == 1:
                lab = labels_d
                length = np.arange(0,len(data[x]),1) 
                pos = 4
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[lab[num]]= data[x][b][c]
                    pos = pos + 6
            
            elif x == 2:
                lab = labels_i
                length = np.arange(0,len(data[x]),1) 
                pos = 0
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[lab[num]]= data[x][b][c]
                    pos = pos + 6
            else:
                lab = labels_i
                length = np.arange(0,len(data[x]),1) 
                pos = 4
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc[lab[num]]= data[x][b][c]
                    pos = pos + 6
    else:
        print('Generando la matriz de datos de electromiografía para marcha en plano')        

        listado = np.arange(0, len(data),1)
        labels_d = [f for f in labels if f.startswith('R')]
        labels_i = [f for f in labels if f.startswith('L')]
        pos = 0
        for x in listado[0:13]:
            
            lab = labels_d
            length = np.arange(0,len(data[x]),1)
            for b in length:
                num = int(b + pos)
                dicc[lab[num]] = data[x][b]
            pos = pos + 7

            

        pos = 0 
        for x in listado[13:26]:
            
            lab = labels_i
            length = np.arange(0,len(data[x]),1)

            for b in length:
                num = int(b+pos)
                dicc[lab[num]] = data[x][b]
            pos = pos + 7
    
    dataframe = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dicc.items() ]))        
    #dataframe = pd.DataFrame.from_dict(dicc)
    print('Matriz generada con éxito')        
                
    return dataframe

"""

def create_dataframe_step(data, labels, info):
    #length = np.arange(0,len(data),1)

    dicc_d = {}
    dicc_i = {}
    if info == 0:
        print('Generando la matriz de datos del sensor inercial marcha en plano')

        labels_d = labels[0:72]
        labels_i = labels[72:]
        #labels_i = labels[72:]
        listado = np.arange(0,len(data),1)
        for x in listado:
            if x == 0:
                lab = labels_d
                length = np.arange(0,len(data[x]),1) 
                pos = 0
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc_d[lab[num]]= data[x][b][c]
                    pos = pos + 6
                    #pos = pos + len(r_lab)
            elif x == 1:
                lab = labels_d
                length = np.arange(0,len(data[x]),1) 
                pos = 4
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc_d[lab[num]]= data[x][b][c]
                    pos = pos + 6
                    #pos = pos + len(r_lab)
            
            elif x == 2:
                lab = labels_i
                length = np.arange(0,len(data[x]),1) 
                pos = 0
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc_i[lab[num]]= data[x][b][c]
                    pos = pos + 6
                    #pos = pos + len(l_lab)
            else:
                lab = labels_i
                length = np.arange(0,len(data[x]),1) 
                pos = 4
                for b in length:
                    length_1 = np.arange(0,len(data[x][b]),1)
                    for c in length_1:
                        num = int(c + pos)
                        dicc_i[lab[num]]= data[x][b][c]
                    pos = pos + 6
                    #pos = pos + len(l_lab)
    else:
        print('Generando la matriz de datos de electromiografía para marcha en plano')        

        listado = np.arange(0, len(data),1)
        #labe = np.array(labels)
        labels_d = [f for f in labels if f.startswith('R')]
        labels_i = [f for f in labels if f.startswith('L')]
        #labels_d = list(labe[r_lab_emg])
        #print(labels_d)
        #labels_i = list(labe[l_lab_emg])
        pos = 0
        #len_emg = len(r_lab_emg)+len(l_lab_emg)
        for x in listado[0:13]:
        #for x in listado[0:len_emg]:    
            
            lab = labels_d
            length = np.arange(0,len(data[x]),1)
            for b in length:
                num = int(b + pos)
                dicc_d[lab[num]] = data[x][b]
            #pos = pos + 7
            pos = pos + len(length)

            

        pos = 0 
        for x in listado[13:26]:
        #for x in listado[len_emg:]:            
            
            lab = labels_i
            length = np.arange(0,len(data[x]),1)

            for b in length:
                num = int(b + pos)
                dicc_i[lab[num]] = data[x][b]
            #pos = pos + 7
            pos = pos + len(length)
    
    dataframe_d = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dicc_d.items() ]))
    dataframe_i = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dicc_i.items() ]))        
    #dataframe = pd.DataFrame.from_dict(dicc)
    
    dataframe = pd.concat([dataframe_d,dataframe_i], axis = 0)
    print('Matriz generada con éxito')        
                
    return dataframe

"""
