# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:10:23 2021

@author: cjove
"""


import numpy as np
from scipy.integrate import cumtrapz
import math as math
import itertools

"""
def min_max_mean_std(data, cycles, index_steps):
    length = np.arange(0,len(data),1)
    minimums = []
    maximums = []
    means = []
    stds = []
    for i in length:
        minimum = []
        maximum = []
        mean = []
        std = []
        length_2 = np.arange(0,len(index_steps[i]),1)
        for x in length_2:
            segment = data[i][index_steps[i][x][0]:index_steps[i][x][-1]]
            min_seg = min(segment)
            max_seg = max(segment)
            mean_seg = np.mean(segment)
            std_seg = np.std(segment)
            minimum.append(min_seg)
            maximum.append(max_seg)
            mean.append(mean_seg)
            std.append(std_seg)
        minimums.append(minimum)
        maximums.append(maximums)
        means.append(mean)
        stds.append(std)
        
    return(minimums,maximums,means,stds)
"""
def min_max_mean_std(data, cycles):
    length = np.arange(0,len(data),1)
    minimums = []
    maximums = []
    means = []
    stds = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:      
            min_seg = min(segment)
            max_seg = max(segment)
            mean_seg = np.mean(segment)
            std_seg = np.std(segment)
            minimums.append(min_seg)
            maximums.append(max_seg)
            means.append(mean_seg)
            stds.append(std_seg)
        else:
            continue

    return(minimums,maximums,means,stds)


#titi = min_max_mean_std(acc_filt['Right_Shank_Gy'], det_sts)        

def initial_final(data, cycles):
    length = np.arange(0,len(data),1)
    initials = []
    finals = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:   
            init_seg = segment[0]
            fin_seg = segment[-1]
                
            initials.append(init_seg)
            finals.append(fin_seg)
        else:
            continue
        #initials.append(initial)
        #finals.append(final)
        
    return(initials, finals)

#titi2 = initial_final(acc_filt['Right_Shank_Gy'], det_sts) 



"""
Una p??gina interesante para obtener todas las features caracter??sticas de la
EMG
http://www.psgminer.com/help/emg_features__.htm

A partir de aqu?? se aplica s??lo a los datos de EMG
"""

def mean_absolute_value(data, cycles):
    length = np.arange(0,len(data),1)
    mavs = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:  
            mav_segment = np.mean(abs(segment))
            mavs.append(mav_segment)
        else:
            continue
        #mavs.append(mav)
        
    return(mavs)

#titi3 = mean_absolute_value(acc_filt['Right_Shank_Gy'], det_sts)

def simple_square_integral(data, cycles):
    length = np.arange(0, len(data), 1)
    ssis = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            squares = []
            for y in segment:
                square = y**2
                squares.append(square)
            ssi_segment = np.sum(squares)
            ssis.append(ssi_segment)
        else:
            continue
        #ssis.append(ssi)
        
    return(ssis)

#titi4 = simple_square_integral(emg_filt['Right_TA'], gait_d)


def variance(data, cycles):
    length = np.arange(0, len(data), 1)
    variances = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            squares = []
            for y in segment:
                square = y**2
                squares.append(square)
            variance_segment = np.sum(squares)/(len(squares)-1)
            variances.append(variance_segment)
        else:
            continue
        #variances.append(variance)
        
    return(variances)

#titi5 = variance(emg_filt['Right_TA'], gait_d)

def root_mean_square(data, cycles):
    length = np.arange(0, len(data), 1)
    rmss = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            squares = []
            for y in segment:
                square = y**2
                squares.append(square)
            rms_segment = math.sqrt((np.sum(squares)/(len(squares))))
            rmss.append(rms_segment)
        else:
            continue
        #rmss.append(rms)
        
    return(rmss)

#titi6 = root_mean_square(emg_filt['Right_TA'], gait_d)

def waveform_length(data, cycles):
    length = np.arange(0,len(data),1)
    wfls = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            length_4 = np.arange(0,len(segment),1)
            sub = 0
            for j in length_4[0:-1]:
                subs = abs(segment[j+1]-segment[j])
                sub = sub +subs
            wfls.append(sub)
        else:
            continue
        #wfls.append(wfl)   

    return(wfls)             
    
#titi4 = waveform_length(emg_filt['Right_TA'], gait_d)  

"""
This one is really characteristic of emg, not much sense on applying
it to other signals (IMU)
Code from: https://stackoverflow.com/questions/2936834/python-counting-sign-changes

Zero crossing, slope sign change and willison amplitude need threshold and i'm
not fixing it --> FIX
"""       
def zero_crossing(data, cycles):
    length = np.arange(0,len(data),1)
    zcs = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            zz = len(list(itertools.groupby(segment, lambda segment: segment > 0))) - (segment[0] > 0)
            zcs.append(zz)
        else:
            continue
        #zcs.append(zc)   

    return(zcs) 

#titi7 = zero_crossing(emg_filt['Right_TA'], gait_d)         
    
def slope_sign_changes(data, cycles):
    length = np.arange(0,len(data),1)
    sscs = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            seg_dif = np.diff(segment)
            zz = len(list(itertools.groupby(seg_dif, lambda seg_dif: seg_dif > 0))) - (seg_dif[0] > 0)
    
            sscs.append(zz)
        else:
            continue
        #sscs.append(ssc)   

    return(sscs)   

#titi8 = slope_sign_changes(emg_filt['Right_TA'], gait_d)
    
"""
https://www.researchgate.net/publication/263765853_EMG_Feature_Extraction_for_Tolerance_of_White_Gaussian_Noise
This article shows that 5milivolts is optimal
"""    
    
def willison_amplitude(data, cycles, thres = 0.005):
    length = np.arange(0,len(data),1)
    was = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            length_4 = np.arange(0,len(segment),1)
            w = []
            for j in length_4[0:-1]:
                subs = abs(segment[j]-segment[j+1])
                if subs > thres:
                    w.append(subs)
                else: 
                    continue
            was.append(len(w))
        else:
            continue
        #was.append(wa)   

    return(was)  

#titi9 = willison_amplitude(emg_filt['Right_TA'], gait_d, 0.005)

"""
Here we start with frequency domain features
"""
"""
Attempt to create extract the coefficientes of a sixth-order autoregressive 
model
To start well I first read the documentation available in wiki
https://en.wikipedia.org/wiki/Autoregressive_model    
https://www.youtube.com/watch?v=4O9Rkzm8Q5U

Definition: An autoregression model is a linear regression model that uses 
lagged variables as input variables.

https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
"""
#from statsmodels.tsa.ar_model import AutoReg
#from sklearn.metrics import mean_squared_error

"""
Pending
"""
from scipy.signal import periodogram
from scipy.integrate import cumtrapz
from numpy import linspace, where

#Viene de biosignal plux
def median_frequency(data, cycles, fs= 1000):
    length = np.arange(0,len(data),1)
    medfs = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            freqs, power = periodogram(segment,fs)
            area_freq = cumtrapz(power, freqs, initial=0)
            total_power = area_freq[-1]
            medfs.append(freqs[where(area_freq >= total_power / 2)[0][0]])
        else:
            continue
        #medfs.append(medf)
          

    return(medfs)
    

#titi10 = median_frequency(emg_filt['Right_TA'], gait_d)  

#Revisar
def mean_frequency(data, cycles, fs= 1000):
    length = np.arange(0,len(data),1)
    meafs = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            freqs, power = periodogram(segment,fs)
            #plt.semilogy(freqs, power)
            #plt.show()
            area_freq = cumtrapz(power, freqs, initial=0)
            average_power = area_freq[-1]/len(freqs)
            meafs.append(freqs[where(area_freq >= average_power)[0][0]])
        else: 
            continue
        #meafs.append(meaf)
          

    return(meafs)

#titi11 = mean_frequency(emg_filt['Right_TA'], gait_d)  


from scipy import stats   

def shannon_entropy(data, cycles):
    length = np.arange(0,len(data),1)
    ses = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            unique_elements, counts_elements = np.unique(segment, return_counts=True)
            entropy = stats.entropy(counts_elements)
            ses.append(entropy)
        else:
            continue
        #ses.append(se)
        
    return ses

"""
https://github.com/raphaelvallat
"""
import antropy as ant    
def spectral_entropy(data, cycles, fs = 1000):
    length = np.arange(0,len(data),1)
    spes = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            spect_entropy = ant.spectral_entropy(segment, fs, method='welch', normalize=True)
            spes.append(spect_entropy)
        else:
            continue
        #spes.append(spe)
        
    return spes

#titi12 = spectral_entropy(emg_filt['Right_TA'], gait_d)  

def singlevaluedecomp_entropy(data, cycles, fs = 1000):
    length = np.arange(0,len(data),1)
    svdes = []
    for i in length:
        segment = data[i][cycles[i][0]:cycles[i][1]]
        if len(segment) != 0:
            svd_entropy = ant.svd_entropy(segment, normalize=True)
            svdes.append(svd_entropy)
        else:
            continue
        #svdes.append(svde)
        
    return svdes
    
#titi13 = singlevaluedecomp_entropy(emg_filt['Right_TA'], gait_d)
#wavelet

"""
Article: feature extraction of the first difference of emg time series for emg
pattern recognition
In the article above we have the formula for everything done so for in this
variable_extraction.py
https://pysiology.readthedocs.io/en/latest/electromiography.html
"""



    
    
    
    
