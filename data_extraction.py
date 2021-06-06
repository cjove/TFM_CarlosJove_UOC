# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:32:11 2021

@author: cjove
"""

"""
This will not be part of the app, as it is the way of applying to the raw data
the filtering we use in mDurance
Docs:
"""

import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter
import os
import signal_processing as sp

def data_extraction(dir_dataset, names):
    print('Empezando la importación de los datos')
    muscle_names = names[30:44]
    acc_names = names[0:30]
    gon_names = names[44:-1]
    #dataframe_test = pd.DataFrame(columns= muscle_names)
    #dir_dataset = 'C:/Users/cjove/Desktop/TFM/Dataset/'
    #results = []
    #name = []
    circuits = np.arange(1,51,1)
    mus = []
    acc = []
    goni = []
    index_steps = []
    
    with os.scandir(dir_dataset) as entries:
        for entry in entries:
            semi_dir = str(str(dir_dataset)+entry.name+"/"+entry.name+
                               "/Raw/"+entry.name+"_Circuit_")
            for number in circuits:
                if number < 10:
                    rep = "00"+str(number)
                else:
                    rep = "0"+str(number)
                try:
                    test = pd.read_csv(semi_dir+str(rep)+
                                   "_raw.csv")
                    print(semi_dir+str(rep)+
                                   "_raw.csv")
                    musc =[]
                    for muscle in muscle_names:
                        m_filtered = sp.butter_bandpass_filter(np.asarray(test[muscle]))
                        #print(len(filtered))
                        musc.append(m_filtered)
                        #name.append(semi_dir+str(rep)+
                        #          "_raw.csv")
                    acce = []
                    for accel in acc_names:
                        a_filtered = sp.butter_lowpass_filter(np.asarray(test[accel]))
                        acce.append(a_filtered)
                    gonio = []
                    for gon in gon_names:
                        g_filtered = sp.butter_lowpass_filter(np.asarray(test[gon]),lowcut = sp.filter_lowcut_gon)
                        gonio.append(g_filtered)
                    ind = test['Mode'][test['Mode']==1].index
                    ind_div = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(ind), lambda x: x[0]-x[1])]
                    index_steps.append(ind_div)
                    mus.append(musc)
                    acc.append(acce)
                    goni.append(gonio)                    
                except Exception:
                    pass
        
    
    emg_filt = pd.DataFrame(mus, columns = muscle_names)
    acc_filt = pd.DataFrame(acc, columns = acc_names)
    gon_filt = pd.DataFrame(goni, columns = gon_names)
    print('Importación terminada')
    return emg_filt, acc_filt, gon_filt, acc_names, muscle_names, index_steps

"""
Remember index steps are the reference to make sure we are in the walking segment
"""