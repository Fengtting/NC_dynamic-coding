# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:41:44 2021

@author: 18110
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                                  
import numpy.fft as fft
import os


zero_before = 330
zero_after = 1330
bin_size = 20
N = (zero_after-zero_before)/bin_size
bg_before = -300
bg_after = 0
M = (bg_after-bg_before)/bin_size

dir = "H:/Biomotion/experientdata/data/BT_leftBrain/FST_Analysis/trans_adding/" 

modulation = pd.DataFrame([])
modulationCon = pd.DataFrame([])
Spikes_count = pd.Series([])
Spikes_bg_count = pd.Series([])
Spikes_net_count = pd.Series([])
Spikes_net_max = pd.Series([])
Net_response = pd.DataFrame([])
Spikes_firing_rate = pd.DataFrame([])
Spikes_bg = pd.DataFrame([])
F1 = pd.DataFrame([])

f_index = []
for root,dirs,files in os.walk(dir):    
    for file in files:
        f = file[3:23]
        current_filename = root + file
        Spikes = pd.read_excel(current_filename)###spikes of every bin
#        spikes_count - spikes.sum()/50
#        spikes_avg = spikes_count.loc[1:8].mean()  ###mean firing rate 
        spikes = Spikes.tail(N)
        spikes_bg = Spikes.head(M)
#        spikes_count = spikes.sum()/N  ###mean firing rate 
        spikes_count = spikes.mean()  ###mean firing rate 
        spikes_count_avg = spikes_count.loc[1:8].mean()  ###average mean firing rate
#        spikes_bg_count = spikes_bg.sum()/M   ##background activity
        spikes_bg_count = spikes_bg.mean()          ##background activity
        spikes_bg_count_avg = spikes_bg_count.loc[1:8].mean()  #average background activity across 1-8 condition
        spikes_count_net = abs(spikes_count - spikes_bg_count_avg) ##net response of 1-8 condition
        spikes_count_net_avg = spikes_count_net.loc[1:8].mean()   ##average net response across 1-8
        spikes_count_net_max = spikes_count_net.loc[1:8].max()  ##max net respons across 1-8
#        spikes_count = spikes_count - spikes_bg_avg
        spikes_avg_control = spikes_count.loc[9:12].mean()
        ###friour transform
        data = pd.DataFrame(spikes)
        fft_data = pd.DataFrame([])
        pows = pd.DataFrame([])
        for k in data.columns:
            cond_data = data[k]
            fft_data[k] = fft.fft(cond_data)
            pows[k] = np.abs(fft_data[k])
            pows[k] = pows[k]/N     ##normalization
#the frequency after fourier transform
        bins = np.arange(zero_before,zero_after+bin_size,bin_size)
        bins_mean = (bins[1:]+bins[:-1])/2.0
        freqs = fft.fftfreq(bins_mean.size,(bins_mean[1]-bins_mean[0])/1000)
        Freqs = fft.fftfreq(bins_mean.size,(bins_mean[1]-bins_mean[0])/1000)
        
#caculate the walking direction index        
        for i,v in enumerate(freqs):
            if v == 2:
                a = i

        modulation_index = {}
        pow_2Hz = pows.loc[a]*2*2
        for i in data.columns:
            modulation_index[i] = pow_2Hz.loc[i]/spikes_count_avg  ###(F1/F0mean8)
        modulation_index = pd.DataFrame(modulation_index,columns=data.columns,index=[f])
        modulation = modulation.append(modulation_index)
        ###F1 
        F1_dict = {}
        for i in data.columns:
            F1_dict[i] = pow_2Hz.loc[i]###F1
        F1_index = pd.DataFrame(F1_dict,columns=data.columns,index=[f])
        F1 = F1.append(F1_index)
        ####average net response
        spikes_net_count_index = [spikes_count_net_avg]
        Spikes_net_count_index = pd.Series(spikes_net_count_index,index=[f])
        Spikes_net_count = Spikes_net_count.append(Spikes_net_count_index)
        ####max net response
        spikes_net_max_index = [spikes_count_net_max]
        Spikes_net_max_index = pd.Series(spikes_net_max_index,index=[f])
        Spikes_net_max = Spikes_net_max.append(Spikes_net_max_index)
        ##average firing rate
        spikes_count_index = [spikes_count_avg]
        Spikes_count_index = pd.Series(spikes_count_index,index=[f])
        Spikes_count = Spikes_count.append(Spikes_count_index)
        ####average bachground activity
        spikes_bg_count_index = [spikes_bg_count_avg]
        Spikes_bg_count_index = pd.Series(spikes_bg_count_index,index=[f])
        Spikes_bg_count = Spikes_bg_count.append(Spikes_bg_count_index)
        ###net response
        net_response_index = {}
        for i in data.columns:
            net_response_index[i] = abs(spikes_count_net[i])   ###net response
        Net_response_index = pd.DataFrame(net_response_index,columns=data.columns,index=[f])
        Net_response = Net_response.append(Net_response_index)
        ####spikes firing rate
        spikes_firing_rate_dict = {}
        for i in data.columns:
            spikes_firing_rate_dict[i] = abs(spikes_count[i])   
        Spikes_firingRate_index = pd.DataFrame(spikes_firing_rate_dict,columns=data.columns,index=[f])
        Spikes_firing_rate = Spikes_firing_rate.append(Spikes_firingRate_index)
        ####background activity
        spikes_bg_dict = {}
        for i in data.columns:
            spikes_bg_dict[i] = abs(spikes_bg_count[i])   
        Spikes_bg_index = pd.DataFrame(spikes_bg_dict,columns=data.columns,index=[f])
        Spikes_bg = Spikes_bg.append(Spikes_bg_index)


        for I,V in enumerate(Freqs):
            if V == 1:
                b = I
        modulation1_index = {}
        pow_1Hz = pows.loc[b]*2*2
        for i in data.columns:
            modulation1_index[i] = pow_1Hz.loc[i]/spikes_avg_control 

        modulation1_index = pd.DataFrame(modulation1_index,columns=data.columns,index=[f])
        modulationCon = modulationCon.append(modulation1_index)

modulation = modulation.loc[:,[1,2,3,4,5,6,7,8]]
modulationCon = modulationCon.loc[:,[9,10,11,12]]
Modulation = pd.concat([modulation,modulationCon],axis=1)

Modulation.dropna(axis=0,how='any',inplace=True)

###

writer=pd.ExcelWriter("H:/Biomotion/experientdata/data/BT_leftBrain/FST_Analysis/" + "MI_F1F0_bin20.xlsx")
##################
Modulation.to_excel(writer,'MI')
F1.to_excel(writer,'F1')
Spikes_firing_rate.to_excel(writer,'Firing rate')
Net_response.to_excel(writer,'Net response')
Spikes_bg.to_excel(writer,'Background response')
Spikes_count.to_excel(writer,'Average firing rate')
Spikes_net_count.to_excel(writer,'Average net response')
Spikes_bg_count.to_excel(writer,'Average background resposne')
Spikes_net_max.to_excel(writer,'Max net response')
writer.save()