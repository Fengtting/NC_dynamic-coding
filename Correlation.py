# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:42:28 2021

@author: 18110
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:58:39 2020

@author: Cherry
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from decimal import Decimal


#########################################################################FST
file1 = "H:/Biomotion/experientdata/data/HD/Analysis/" + "BioIndex.xlsx"
file2 = "H:/Biomotion/experientdata/data/BT_leftBrain/Analysis/" + "BioIndex.xlsx"
biomotion1 = pd.DataFrame(pd.read_excel(file1))
biomotion2 = pd.DataFrame(pd.read_excel(file2))
biomotion = pd.concat([biomotion1,biomotion2],axis=0)
#
file3 = "H:/Biomotion/experientdata/data/BT_leftBrain/Analysis/" + "OpticFlow_withSpon.xlsx"
file4 = "H:/Biomotion/experientdata/data/HD/Analysis/" + "OpticFlow_withSpon.xlsx"
direction1 = pd.DataFrame(pd.read_excel(file3))
direction2 = pd.DataFrame(pd.read_excel(file4))
direction = pd.concat([direction1,direction2],axis=0)
#######################################################################
df = pd.concat([biomotion1,direction1],axis=1)
df = df.dropna(axis=0,how='any')
df_copy = df
df = df.loc[df['formIndex']>0]
n = df.shape[0]
####rad
r1,p1 = stats.pearsonr(abs(df['RadiationIndex']),df['formIndex'])
r1,p1 = stats.pearsonr(abs(df['RadiationIndex']),df['inversionIndex'])
r1,p1 = stats.pearsonr((df['RadiationIndex']),abs(df['walkingDirIndex']))
####rot
r1,p1 = stats.pearsonr(abs(df['RotationIndex']),df['formIndex'])
r1,p1 = stats.pearsonr(abs(df['RotationIndex']),df['inversionIndex'])
r1,p1 = stats.pearsonr(abs(df['RotationIndex']),abs(df['walkingDirIndex']))
#############dir VS walking direction

R1 = "{:.3f}".format(r1)
P1 = "{:.5f}".format(p1)
R2 = "{:.3f}".format(r2)
P2 = "{:.5f}".format(p2)
R3 = "{:.3f}".format(r3)
P3 = "{:.5f}".format(p3)
plt.figure(figsize=(9,9))  #(5,10)
#####form
g1 = sns.regplot(df['formIndex'],df['RadiationIndex'],color='#DA2727')  ##DA2727
g1 = sns.regplot(df['formIndex'],df['RotationIndex'],color='#DA2727',marker='o')    
#######inversion
g1 = sns.regplot(df['inversionIndex'],df['RadiationIndex'],color='#006837')
g1 = sns.regplot(df['inversionIndex'],df['RotationIndex'],color='#006837',marker='o')
###walking direction
g1 = sns.regplot(abs(df['walkingDirIndex']),df['RadiationIndex'],color='#4171C7')
g1 = sns.regplot(abs(df['walkingDirIndex']),df['RotationIndex'],color='#4171C7',marker='o')

plt.xticks([0,0.5,1.0,1.5],fontsize=20)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=20))
g1.legend(title='r: {0}\np: {1}\nN:{2}'.format(R1,P1,n))
#g1.legend(title='r1: {0}\np1: {1}\nr2: {2}\np2: {3}\nr3: {4}\np3: {5}\nN:{6}'.format(R1,P1,R2,P2,R3,P3,n))
#g1.title('form index>0')
#plt.savefig('D:/Biomotion/experientdata/data/interim/MT/Corr_WKVSdir_.eps',dpi=300)
