'''
Code to calculate correlation coefficients between timeseries of NAO values centered on surface impact date. Code also calculates effective sample size (ESS) and which correlation coefficients are significant based on ESS
'''

import numpy as np
import math
import os
import pickle
import pandas as pd
import xarray as xr 
import scipy.stats 
from statsmodels.tsa.stattools import acf

#define function to find filepaths to individual CMIP6 model files of mean NAO across all SSW events, centered on surface impact date
def list_files(dir):
    r=[]
    exclude_prefixes=('_','.')
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath=root + os.sep +name  
            if filepath.endswith("NAO_surface_all_events_mean.nc"):
                if not name.startswith(exclude_prefixes): 
                    r.append(os.path.join(root,name))
    return r

filepaths=list_files("/path_to_files/")
filepaths.sort()

#model labels for pandas DataFrame
models=['CCCma','HAMMOZ', 'HadGEM','INM','MIROC','MPI_HR', 'MPI_LR', 'NCAR', 'UKESM', 'WACCM']

#open reanalysis NAO file
DS2=xr.open_dataset('/path_to_file/file.nc')
ERA_NAO=DS2.msl.squeeze()
DS2.close()

#acf for ERA
ERA_acf=acf(ERA_NAO.values,nlags=60,fft=False)

#calculate correlation coefficients for ERA/CMIP model, acf of CMIP model
correls=[]
acfs=[]
for i in range(0, len(filepaths)):
    DS=xr.open_dataset(filepaths[i])
    NAO=DS.psl.squeeze()
    DS.close()
    NAO_cor_coef, NAO_pval=scipy.stats.pearsonr(NAO.values, ERA_NAO.values)
    correls.append(NAO_cor_coef)
    model_acf=acf(NAO.values,nlags=60,fft=False)
    acfs.append(model_acf)

cors_arr=np.array(correls)
acfs_arr=np.array(acfs).reshape(len(filepaths),61)
acfs_arr=acfs_arr[:,1:61]
correls_df=pd.DataFrame(correls,columns=['NAO'],index=models)
print(correls_df)

#calculate effective sample size
tau=np.arange(1,61)
T=61

ERA_acf=ERA_acf[1:61]
ESS_models=[]
p_vals=[]
sig_vals=[]
for i in range(0,len(acfs_arr)):
	denoms=[]
	for j in range(0,60):
		autocorr=ERA_acf[j]*acfs_arr[i][j]
		denom=(1-(tau[j]/T))*autocorr
		denoms.append(denom)
	sum_denoms=sum(denoms)*2
	ESS=T/sum_denoms
	ESS=np.round(ESS,decimals=0)
	ESS_models.append(ESS)	
	dist=scipy.stats.beta(ESS/2 - 1, ESS/2 - 1, loc=-1, scale=2)
	p=2*dist.cdf(-abs(cors_arr[i]))
	p_vals.append(p)
	if p < 0.1:
		sig_vals.append(cors_arr[i])
	else:
		sig_vals.append(np.nan)

print('ESS models:',ESS_models)
print('p vals:', p_vals)
print('sig vals:', sig_vals)



















