# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:41:05 2021

@author: 18110
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:23:02 2021

@author: 18110
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:37:11 2021

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


#file1 = "D:/Biomotion/experientdata/data/HD/Analysis/" + "MI_F1F0_bin20.xlsx"
#file2 = "D:/Biomotion/experientdata/data/BT_leftBrain/Analysis/" + "MI_F1F0_bin20.xlsx"
#mst1 = pd.read_excel(file1)
#mst2 = pd.read_excel(file2)
#mst = pd.concat([mst1,mst2],axis=0)
file1 = "G:/Paper1/the manuscript/NC repository/data/" + "MI_all.xlsx"
mst = pd.read_excel(file1)
#file1 = "G:/Paper1/the manuscript/NC repository/data/" + "MI_all.xlsx"
#mst = pd.read_excel(file1)
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


#########################################################################################################################################################
form = pd.DataFrame(index=mst.index,columns=mst.columns)
inversion = pd.DataFrame(index=mst.index,columns=mst.columns)
walkingDirection = pd.DataFrame(index=mst.index,columns=mst.columns)
nonCoding = pd.DataFrame(index=mst.index,columns=mst.columns)
sampling = pd.DataFrame(index=mst.index,columns=mst.columns)
bio_idx = pd.DataFrame(index=mst.index,columns=['form','f1','f3','f5','f7','inversion','inv1','inv5','WK','wk1','wk3','max'])
pairMax = pd.DataFrame(index=mst.index,columns=['form_max','inversion_max','WK_max'])
###############################################################################################################
####class into 4 type(form cells, inversion cells, WK cells, and cells that do not encoding any information)
for idx in mst.index:
    form_index = max(abs(mst.loc[idx,1]-mst.loc[idx,2]),abs(mst.loc[idx,3]-mst.loc[idx,4]),abs(mst.loc[idx,5]-mst.loc[idx,6]),abs(mst.loc[idx,7]-mst.loc[idx,8]))
    if abs(mst.loc[idx,1]-mst.loc[idx,2]) == form_index:
        pairMax.loc[idx,'form_max'] = max(mst.loc[idx,1],mst.loc[idx,2])
    if abs(mst.loc[idx,3]-mst.loc[idx,4]) == form_index:
        pairMax.loc[idx,'form_max'] = max(mst.loc[idx,3],mst.loc[idx,4])
    if abs(mst.loc[idx,5]-mst.loc[idx,6]) == form_index:
        pairMax.loc[idx,'form_max'] = max(mst.loc[idx,5],mst.loc[idx,6])
    if abs(mst.loc[idx,7]-mst.loc[idx,8]) == form_index:
        pairMax.loc[idx,'form_max'] = max(mst.loc[idx,7],mst.loc[idx,8])
    
    inversion_index = max(abs(mst.loc[idx,1]-mst.loc[idx,3]),abs(mst.loc[idx,5]-mst.loc[idx,7]))
    if abs(mst.loc[idx,1]-mst.loc[idx,3]) == inversion_index:
        pairMax.loc[idx,'inversion_max'] = max(mst.loc[idx,1],mst.loc[idx,3])
    else:
        pairMax.loc[idx,'inversion_max'] = max(mst.loc[idx,5],mst.loc[idx,7])
    WK_index = max(abs(mst.loc[idx,1]-mst.loc[idx,5]),abs(mst.loc[idx,3]-mst.loc[idx,7]))
    if abs(mst.loc[idx,1]-mst.loc[idx,5]) == WK_index:
        pairMax.loc[idx,'WK_max'] = max(mst.loc[idx,1],mst.loc[idx,5])
    else:
        pairMax.loc[idx,'WK_max'] = max(mst.loc[idx,3],mst.loc[idx,7])
    bio_idx.loc[idx,'form'] = form_index
    bio_idx.loc[idx,'f1'] = abs(mst.loc[idx,1]-mst.loc[idx,2])
    bio_idx.loc[idx,'f3'] = abs(mst.loc[idx,3]-mst.loc[idx,4])
    bio_idx.loc[idx,'f5'] = abs(mst.loc[idx,5]-mst.loc[idx,6])
    bio_idx.loc[idx,'f7'] = abs(mst.loc[idx,7]-mst.loc[idx,8])
    bio_idx.loc[idx,'inversion'] = inversion_index
    bio_idx.loc[idx,'inv1'] = abs(mst.loc[idx,1]-mst.loc[idx,3])
    bio_idx.loc[idx,'inv5'] = abs(mst.loc[idx,5]-mst.loc[idx,7])
    bio_idx.loc[idx,'WK'] = WK_index
    bio_idx.loc[idx,'wk1'] = abs(mst.loc[idx,1]-mst.loc[idx,5])
    bio_idx.loc[idx,'wk3'] = abs(mst.loc[idx,3]-mst.loc[idx,7])
    bio_idx.loc[idx,'max'] = max(form_index,inversion_index,WK_index)
#####################class cells according to bio index
    threshod = 0.5
    if form_index == max(form_index,inversion_index,WK_index):
        if form_index < threshod:
            nonCoding.loc[idx,:] = mst.loc[idx,:]
        else:
            form.loc[idx,:] = mst.loc[idx,:]
    if inversion_index == max(form_index,inversion_index,WK_index):
        if inversion_index < threshod:
            nonCoding.loc[idx,:] = mst.loc[idx,:]
        else:
            inversion.loc[idx,:] = mst.loc[idx,:]
    if WK_index == max(form_index,inversion_index,WK_index):
        if WK_index < threshod:
            nonCoding.loc[idx,:] = mst.loc[idx,:]
        else:
            walkingDirection.loc[idx,:] = mst.loc[idx,:]
                

#################################
Form = form.dropna(axis=0,how='any')
Inversion = inversion.dropna(axis=0,how='any')
WalkingDirection = walkingDirection.dropna(axis=0,how='any')
NonCoding = nonCoding.dropna(axis=0,how='any')
##
writer=pd.ExcelWriter("H:/Biomotion/experientdata/data/NC/" + "NonsensitiveCell_maxdiff.xlsx")
NonCoding.to_excel(writer,'Sheet1')
writer.save()
###
writer=pd.ExcelWriter("H:/Biomotion/experientdata/data/NC/" + "FormCell_maxdiff.xlsx")
Form.to_excel(writer,'Sheet1')
writer.save()
####
writer=pd.ExcelWriter("H:/Biomotion/experientdata/data/NC/" + "InversionCell_maxdiff.xlsx")
Inversion.to_excel(writer,'Sheet1')
writer.save()
####
writer=pd.ExcelWriter("H:/Biomotion/experientdata/data/NC/" + "WalkingDirectionCell_maxdiff.xlsx")
WalkingDirection.to_excel(writer,'Sheet1')
writer.save()
#############################################################################################################################################