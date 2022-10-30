# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:05:29 2021

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


file2 = "G:/Paper1/the manuscript/NC repository/data/" + "MI_all.xlsx"

mst = pd.read_excel(file2)
#################################################
mean_mst = mst.mean()
std_mst = mst.std()
sem_mst = std_mst/(math.sqrt(mst.shape[0]))

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

## Form
####max diff between intact and scrambled
for idx in mst.index:
    form_index = max(abs(mst.loc[idx,1]-mst.loc[idx,2]),abs(mst.loc[idx,3]-mst.loc[idx,4]),abs(mst.loc[idx,5]-mst.loc[idx,6]),abs(mst.loc[idx,7]-mst.loc[idx,8]))
    if abs(mst.loc[idx,1]-mst.loc[idx,2]) == form_index:
        preference[idx] = mst.loc[idx,1]
        Nonpreference[idx] = mst.loc[idx,2]
    if abs(mst.loc[idx,3]-mst.loc[idx,4]) == form_index:
        preference[idx] = mst.loc[idx,3]
        Nonpreference[idx] = mst.loc[idx,4]
    if abs(mst.loc[idx,5]-mst.loc[idx,6]) == form_index:
        preference[idx] = mst.loc[idx,5]
        Nonpreference[idx] = mst.loc[idx,6]
    if abs(mst.loc[idx,7]-mst.loc[idx,8]) == form_index:
        preference[idx] = mst.loc[idx,7]
        Nonpreference[idx] = mst.loc[idx,8]
form_prefer = preference
form_nonprefer = Nonpreference
form_diff = preference - Nonpreference
mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
print('#######################form')
print(mean_prefer)
print(mean_Nonprefer)
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)

p = [p]
plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['#DA2727','#E1A5B3'], capsize=5)  
plt.xticks(xtik,['intact','scramble'],fontsize=18)
#plt.xticks(xtik,['intact_up_right','scramble_up_right'],fontsize=12)
plt.yticks([0,0.5,1.0,1.5],fontsize=15)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.tick_params(bottom=False)
plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
plt.xlim([0.5,3.2])
plt.ylim([0,1.5])
plt.ylabel('MI',fontsize=15)   #Average firing rate
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/form_all.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/form_all.eps',dpi=300)

##max diff between up and down
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,3]) >= abs(mst.loc[idx,5]-mst.loc[idx,7]):
        preference[idx] = mst.loc[idx,1]
        Nonpreference[idx] = mst.loc[idx,3]
        d_pre_nonpre[idx] = mst.loc[idx,1] - mst.loc[idx,3]
    else:
        preference[idx] = mst.loc[idx,5]
        Nonpreference[idx] = mst.loc[idx,7]
        d_pre_nonpre[idx] = mst.loc[idx,5] - mst.loc[idx,7]
mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
mean_d = d_pre_nonpre.mean()    
t,p = ttest_rel(preference,Nonpreference)
print('#######################up VS down')
print(mean_prefer)
print(mean_Nonprefer)
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))    
sem_d = d_pre_nonpre.std()/(math.sqrt(Nonpreference.size-1))  
t,p = ttest_rel(preference,Nonpreference)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
print(mean_d)
print(sem_d)
p = [p]
plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['#006837','#90B686'], capsize=5)  #hatch='//'
plt.xticks(xtik,['up','down'],fontsize=18)
n = preference.shape[0]
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.yticks([0,0.5,1.0,1.5],fontsize=15)
plt.tick_params(bottom=False)
plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
plt.xlim([0.5,3.2])
plt.ylim([0,1.5])
plt.ylabel('MI',fontsize=15)
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/upVSdown.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/upVSdown.eps',dpi=300)
########################################################################
#####Inversion(max diff between prefer and nonprefer
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,3]) >= abs(mst.loc[idx,5]-mst.loc[idx,7]):
        if mst.loc[idx,1] >= mst.loc[idx,3]:
            preference[idx] = mst.loc[idx,1]
            Nonpreference[idx] = mst.loc[idx,3]
        if mst.loc[idx,1] < mst.loc[idx,3]:
            preference[idx] = mst.loc[idx,3]
            Nonpreference[idx] = mst.loc[idx,1]
    if abs(mst.loc[idx,5]-mst.loc[idx,7]) >= abs(mst.loc[idx,1]-mst.loc[idx,3]):
        if mst.loc[idx,5] >= mst.loc[idx,7]:            
            preference[idx] = mst.loc[idx,5]
            Nonpreference[idx] = mst.loc[idx,7]
        if mst.loc[idx,5] < mst.loc[idx,7]:
            preference[idx] = mst.loc[idx,7]
            Nonpreference[idx] = mst.loc[idx,5]

mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
print('#######################inversion')
print(mean_prefer)
print(mean_Nonprefer)
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
p = [p]

plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['#006837','#90B686'], capsize=5)  #hatch='//'
plt.xticks(xtik,['prefer','nonprefer'],fontsize=18)
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plt.yticks([0,0.5,1.0],fontsize=15)
plt.tick_params(bottom=False)
plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
plt.xlim([0.5,3.2])
plt.ylim([0,2])
plt.ylabel('MI',fontsize=15)
sns.despine()
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/inversion.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/inversion.eps',dpi=300)

 ####Walking Direction
##max diff between prefer and nonprefer walking dierction
for idx in mst.index:
    if abs(mst.loc[idx,1]-mst.loc[idx,5]) >= abs(mst.loc[idx,3]-mst.loc[idx,7]):
        if mst.loc[idx,1] >= mst.loc[idx,5]:
            preference[idx] = mst.loc[idx,1]
            Nonpreference[idx] = mst.loc[idx,5]
        else:
            preference[idx] = mst.loc[idx,5]
            Nonpreference[idx] = mst.loc[idx,1]
    else:
        if mst.loc[idx,3] >= mst.loc[idx,7]:
            preference[idx] = mst.loc[idx,3]
            Nonpreference[idx] = mst.loc[idx,7]
        else:
            preference[idx] = mst.loc[idx,7]
            Nonpreference[idx] = mst.loc[idx,3]    

mean_prefer = preference.mean()
mean_Nonprefer = Nonpreference.mean()
print('#######################walking direction')
print(mean_prefer)
print(mean_Nonprefer)
sem_prefer = preference.std()/(math.sqrt(preference.size)) 
sem_Nonprefer = Nonpreference.std()/(math.sqrt(Nonpreference.size))     
t,p = ttest_rel(preference,Nonpreference)
print(sem_prefer)
print(sem_Nonprefer)
print(p)
print(t)
p = [p]
###plt.subplot(1,3,3)
plt.figure(figsize=(6,5))
plt.bar(x,[mean_prefer,mean_Nonprefer],width=wid,yerr=[sem_prefer,sem_Nonprefer],color = ['royalblue','lightskyblue'], capsize=5)  
plt.xticks(xtik,['perfer','nonperfer'],fontsize=18)
plt.yticks([0,0.5,1.0,],fontsize=15)
plt.tick_params(bottom=False)
plt.text(2.5,0.4,'N: {0}'.format(n),fontsize=15)
plot_sig(Xstart,Xend,[mean_prefer+0.01],[mean_prefer+0.03],p)
#plot_sig(Xstart,Xend,[mean_prefer+0.5],[mean_prefer+1],p)
plt.xlim([0.5,3.2])
plt.ylim([0,2])
sns.despine()
plt.ylabel('MI',fontsize=15)
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/WalkingDirection.eps',dpi=300)
#plt.savefig('G:/Paper2/figure_MT/WalkingDirection.eps',dpi=300)

