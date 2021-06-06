# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:30:00 2021

@author: cjove
"""

"""
Inside this document the different steps of signal processing will be developed

In this case, the article that generated the data extracted the variables with very little processing
"""

# The tests used to see the difference generated to the signal by the filtering of the article are: test and test_proc


"""
Filtering the signal with a 4th order Butterworth between 20-450Hz. This will
be part of the app. 
Docs: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
"""

filter_lowcut = 20
filter_highcut = 450
filter_order = 4
fs = 1000
   
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fs = fs,lowcut = filter_lowcut, 
                           highcut = filter_highcut, order= filter_order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

filter_order_acc = 6
filter_lowcut_acc = 25

""" 
Estos datos provienen todos de la identificación de los pasos del artículo y
se han replicado para el algoritmo de detección de transición de sentado a de 
pie
"""
#filter_order_acc = 1
#filter_lowcut_acc = 6
filter_lowcut_gon = 10
fs = 1000

def butter_lowpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a



def butter_lowpass_filter(data, fs = fs ,lowcut = filter_lowcut_acc, 
                            order= filter_order_acc):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



