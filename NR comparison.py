# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:55:21 2021

@author: 18110
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:19:05 2021

@author: 18110
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:44:58 2020

@author: Cherry
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from pylab import *  
from  scipy.stats import ttest_rel
import matplotlib as mpl
plt.rc('font',family='Arial')
#mpl.rcParams['font.serif'] = ['Times New Roman']
#mpl.rcParams[u'font.sans-serif'] = ['Times New Romans'] #指定默认字体  
#mpl.rcParams['axes.unicode_minus'] = False 

file1 = "G:/Paper1/the manuscript/NC repository/data/" + "FormCell_maxdiff.xlsx"
mst = pd.read_excel(file1)
file2 = "G:/Paper1/the manuscript/NC repository/data/" + "NR_all.xlsx"
fr = pd.read_excel(file2)



x = [1,1.8]
xs = [1.45,2.0]
xtik = [1.0,1.8]
wid=0.45
Xstart = [1.05]
Xend = [1.75]


def plot_sig(xstart,xend,ystart,yend,sig):
    for i in range(len(xstart)):
        x = np.ones((2))*xstart[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.001)
        plt.plot(x,y,label="$y$",color="grey",linewidth=2)

        x = np.arange(xstart[i],xend[i]+0.001,xend[i]-xstart[i])
        y = yend[i]+0*x
        plt.plot(x,y,label="$y$",color="grey",linewidth=2)

        x0 = (xstart[i]+xend[i])/2
        y0=yend[i]
        if sig[i]>0.05:
            plt.annotate(' ', xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16)
        if sig[i]<0.05 and sig[i]>0.01:
            plt.annotate(r'$*$', xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16,color="black")
        if sig[i]<0.01 and sig[i]>0.001:
            plt.annotate(r'$**$', xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16,color="black")
        if sig[i]<0.001:
            plt.annotate(r'$***$', xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                         textcoords='offset points', fontsize=16,color="black")
        x = np.ones((2))*xend[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.001)
        plt.plot(x,y,label="$y$",color="grey",linewidth=2)        
#plot_sig([0.42,1.42],[1.42,2.42],[30,20],[30.8,20.8],[0.01,0.6])

preference = pd.Series(index=mst.index)
Nonpreference = pd.Series(index=mst.index)
d_pre_nonpre = pd.Series(index=mst.index)

#### Form(difference between intact and scramble)
##max diff
for idx in mst.index:
    form_index = max(abs(mst.loc[idx,1]-mst.loc[idx,2]),abs(mst.loc[idx,3]-mst.loc[idx,4]),abs(mst.loc[idx,5]-mst.loc[idx,6]),abs(mst.loc[idx,7]-mst.loc[idx,8]))
    if abs(mst.loc[idx,1]-mst.loc[idx,2]) == form_index:
        preference[idx] = fr.loc[idx,1]
        Nonpreference[idx] = fr.loc[idx,2]
    if abs(mst.loc[idx,3]-mst.loc[idx,4]) == form_index:
        preference[idx] = fr.loc[idx,3]
        Nonpreference[idx] = fr.loc[idx,4]
    if abs(mst.loc[idx,5]-mst.loc[idx,6]) == form_index:
        preference[idx] = fr.loc[idx,5]
        Nonpreference[idx] = fr.loc[idx,6]
    if abs(mst.loc[idx,7]-mst.loc[idx,8]) == form_index:
        preference[idx] = fr.loc[idx,7]
        Nonpreference[idx] = fr.loc[idx,8]
form_prefer = preference
form_nonprefer = Nonpreference
form_diff = preference - Nonpreference
mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print('####################form')
print(mean_prefer)
print(mean_Nonprefer)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
p = [p]
plt.figure(figsize=(6,5))
#plt.figure(figsize=(17,5))
#plt.subplot(1,3,1)
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['red','pink'], capsize=5)  
plt.xticks(xtik,['intact','scramble'],fontsize=18)
plt.yticks([0,5,10,15,20],fontsize=15)
#plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.tick_params(bottom=False)
#plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
plot_sig(Xstart,Xend,[mean_prefer+0.5],[mean_prefer+1],p)
plt.xlim([0.5,3.0])
#plt.ylim([0,0.8])
plt.ylim([0,25])
plt.ylabel('net response',fontsize=15)   #Average firing rate
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/F0_FormF0bar.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/F0_FormF0bar.eps',dpi=300)

########################Inversion (diff between prefer and nonprefer)
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,3]) >= abs(mst.loc[idx,5]-mst.loc[idx,7]):
        if mst.loc[idx,1] >= mst.loc[idx,3]:
            preference[idx] = fr.loc[idx,1]
            Nonpreference[idx] = fr.loc[idx,3]
        else:
            preference[idx] = fr.loc[idx,3]
            Nonpreference[idx] = fr.loc[idx,1]
    else:
        if mst.loc[idx,5] >= mst.loc[idx,7]:
            preference[idx] = fr.loc[idx,5]
            Nonpreference[idx] = fr.loc[idx,7]
        else:
            preference[idx] = fr.loc[idx,7]
            Nonpreference[idx] = fr.loc[idx,5]

mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print('####################inversion')
print(mean_prefer)
print(mean_Nonprefer)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
p = [p]
plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['green','lightgreen'], capsize=5) 
plt.xticks(xtik,['prefer','nonprefer'],fontsize=18)
plt.yticks([0,5,10,15,20],fontsize=15)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.tick_params(bottom=False)
plot_sig(Xstart,Xend,[mean_prefer+0.5],[mean_prefer+1],p)
plt.ylim([0,25])
plt.xlim([0.5,3.0])
plt.ylabel('net response',fontsize=15)
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/F0_InversionF0bar.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/F0_InversionF0bar.eps',dpi=300)
######Inversion(difference between up and down)
##max F0 diff between up and down
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,3]) >= abs(mst.loc[idx,5]-mst.loc[idx,7]):
        preference[idx] = fr.loc[idx,1]
        Nonpreference[idx] = fr.loc[idx,3]
        d_pre_nonpre[idx] = fr.loc[idx,1] - fr.loc[idx,3]
    else:
        preference[idx] = fr.loc[idx,5]
        Nonpreference[idx] = fr.loc[idx,7]
        d_pre_nonpre[idx] = fr.loc[idx,5] - fr.loc[idx,7]
mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
mean_d = d_pre_nonpre.mean()
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))
sem_d = d_pre_nonpre.std()/(math.sqrt(Nonpreference.size-1))     
t,p = ttest_rel(preference,Nonpreference)
print('####################up VS down')
print(mean_prefer)
print(mean_Nonprefer)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
print(mean_d)
print(sem_d)
p = [p]
plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['#006837','#90B686'], capsize=10, hatch='//')  
plt.xticks(xtik,['up','down'],fontsize=18)
plt.yticks([0,5,10,15,20],fontsize=15)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.tick_params(bottom=False)
plot_sig(Xstart,Xend,[mean_prefer+0.5],[mean_prefer+1],p)
plt.ylim([0,25])
plt.xlim([0.5,3.0])
#plt.ylim([0,0.8])
plt.ylabel('net response',fontsize=15)
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/F0_upVSdownF0bar.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/F0_upVSdownF0bar.eps',dpi=300)

 ####Walking Direction(difference between prefer and nonprefer walking direction)
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,5]) >= abs(mst.loc[idx,3]-mst.loc[idx,7]):
        if mst.loc[idx,1] >= mst.loc[idx,5]:
            preference[idx] = fr.loc[idx,1]
            Nonpreference[idx] = fr.loc[idx,5]
        else:
            preference[idx] = fr.loc[idx,5]
            Nonpreference[idx] = fr.loc[idx,1]
    else:
        if mst.loc[idx,3] >= mst.loc[idx,7]:
            preference[idx] = fr.loc[idx,3]
            Nonpreference[idx] = fr.loc[idx,7]
        else:
            preference[idx] = fr.loc[idx,7]
            Nonpreference[idx] = fr.loc[idx,3]    

mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print('####################walking direction')
print(mean_prefer)
print(mean_Nonprefer)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
p = [p]
plt.figure(figsize=(6,5))
#plt.subplot(1,3,3)
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['royalblue','lightskyblue' ], capsize=5)  
plt.xticks(xtik,['perfer','nonperfer'],fontsize=18)
plt.yticks([0,5,10,15,20,25],fontsize=15)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
#plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=15)
plt.tick_params(bottom=False)
#plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
plot_sig(Xstart,Xend,[mean_prefer+0.5],[mean_prefer+1],p)
plt.xlim([0.5,3.0])
#plt.ylim([0,0.8])
plt.ylim([0,25])
sns.despine()
plt.ylabel('net response',fontsize=15)
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/F0_WKF0bar.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/F0_WKF0bar.eps',dpi=300)
#plt.suptitle('max idx')
