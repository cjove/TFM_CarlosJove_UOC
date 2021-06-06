# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:00:56 2021

@author: cjove
"""
import numpy as np
from scipy.signal import find_peaks
import signal_processing as sp

def peaks_gyro(data, height = 0.2,distance = 500, width = 50 , std = 2):
    pos_peaks_gx = []
    neg_peaks_gx = []

    for x in data['Waist_Gx']:
        x = sp.butter_lowpass_filter(x, lowcut = 6, order = 1, fs = 1000)
        height = std*np.std(x)
        peaks,_ = find_peaks(x, height= height, distance=distance, width= width)
        peaks_neg, _ = find_peaks(-x, height = height, distance=distance, width = width)
        
        pos_peaks_gx.append(peaks)
        neg_peaks_gx.append(peaks_neg)
   
    
    return(pos_peaks_gx, neg_peaks_gx)
        
"""
Necessary to check why I haven't written 500 as the distance in the function
In case of doubt distance = 500 
"""
#peaks = peaks_gyro(acc_filt, distance= 500)

"""
Threshold method
"""

def sit_to_stand_threshold(data, std =10, window_thres = 100, window = 25, thres = 0.05):
    puntos = []


    for x in data['Waist_Gx']:
        x = sp.butter_lowpass_filter(x, lowcut = 6, order = 1, fs = 1000)
        data = abs(x)
        #thres = std*np.std(np.asarray(x[0:window_thres]))
      
        array = range(0,len(x),1)
        state = 0
        ptos = []
        for a in array:
            tramo = x[a:a+window]
            if state == 0:
                valor = [abs(i) for i in tramo if abs(i)>thres]
                if len(valor) == window:
                    state = 1
                    ptos.append(a)
                else:
                    continue
            else:
                valor = [abs(i) for i in tramo if abs(i)<thres]
                if len(valor) == window:
                    state = 0
                    ptos.append(a)
                    
                else:
                    continue
        puntos.append(ptos)
    return(puntos)

def sit_to_stand(picos, thres_points):
    positivos = []
    negativos = []
    points = []
    rango = range(0, len(thres_points), 1)
    for i in picos[0]:
        positivos.append(i[0])
    for i in picos[1]:
        negativos.append(i[0])
    for i in rango:
        inicio = 0
        fin = 0
        for x in thres_points[i]:
            if x < positivos[i]:
                inicio = x
            if fin == 0:
                if x > negativos[i]:
                    fin = x
            else:
                continue
        #start.append(inicio)
        #end.append(fin)
        points.append([inicio,fin])
    #return(start,end)
    return(points)
            

#det_sts = sit_to_stand(peaks, thres_method)   