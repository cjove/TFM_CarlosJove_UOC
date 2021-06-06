# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:43:54 2021

@author: cjove
"""
import numpy as np
from scipy.signal import find_peaks
import signal_processing as sp

def gait_cycles(data, index):
    steps = peaks_step(data, index)
    length = np.arange(0,len(steps[0]),1)
    pp = []
    for i in length:
        p = []
        length1 = np.arange(0,len(steps[1][i]),1)
        for x in length1:
            points = []
            length2 = np.arange(1,len(steps[1][i][x]), 1)
            for y in length2:
                points_prev = []
                points_post = []
                val = steps[1][i][x][y]
                for z in steps[0][i][x]:
                    if z < val:
                        points_prev.append(z)
                    else:
                        points_post.append(z)
                if len(points_prev)>=2:
                    start = points_prev[-2]
                else:
                    start = None
                if len(points_post) >= 1:
                    stop = points_post[0]
                else:
                    stop = None
                if start is not None and stop is not None:
                    points.append([start,stop])
                else:
                    continue
            p.append(points)
        pp.append(p)
                
    return(pp)



def peaks_step(data, index, distance = 300, height = 0.5):
    
    #data = sp.butter_lowpass_filter(data, lowcut = 6, order = 1, fs = 1000)
    pos_peaks_gy = []
    neg_peaks_gy = []
    pos_heights_gy = []
    neg_heights_gy = []

    length = np.arange(0,len(data),1)
    
    for x in length:
        test = data[x]
        test = sp.butter_lowpass_filter(test, lowcut = 6, order = 1, fs = 1000)
        ind_length = np.arange(0,len(index[x]),1)
        
        pos_peaks = []
        neg_peaks = []
        pos_heights = []
        neg_heights = []
        
        for y in ind_length:
            peaks, properties_pos = find_peaks(test[index[x][y][0]:index[x][y][-1]], distance = distance, height = height)
            pos_height = properties_pos['peak_heights']
            pos_peaks.append(peaks)
            pos_heights.append(pos_height)
        for y in ind_length:
            peaks_neg, properties_neg = find_peaks(-test[index[x][y][0]:index[x][y][-1]], distance = distance, height = height)
            neg_height = properties_neg['peak_heights']
            neg_peaks.append(peaks_neg)
            neg_heights.append(neg_height)
            
            
        pos_peaks_gy.append(pos_peaks)
        neg_peaks_gy.append(neg_peaks) 
        pos_heights_gy.append(pos_heights)
        neg_heights_gy.append(neg_heights)
        
    return(pos_peaks_gy, neg_peaks_gy, pos_heights_gy, neg_heights_gy)           
            


def invertir(data, index):
    step = peaks_step(data, index)
    length = np.arange(0,len(step[0]),1)
    datos = []
    for i in length:
        length2 = np.arange(0,len(step[3][i]),1)
        for x in length2:
            x = 0
            try:
                if np.max(step[2][i][x]) > np.max(step[3][i][x]):
                    x = x + 1
                else:
                    continue
            except:
                continue
        if x != 0:
            datos.append(-data[i])
        else:
            datos.append(data[i])
    return(datos)
