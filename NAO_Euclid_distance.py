'''
    This code loads already processed data for the NAO from ERA and CMIP6 models, and processes
    the data to calculate Euclidean distances and saves as pandas DataFrame


'''

import numpy as np
import math
import os
import pickle
import pandas as pd
import xarray as xr 
import scipy.stats 
from matplotlib import pyplot as plt


#define function to find filepaths to CMIP6 data
def list_files(dir):
    r=[]
    exclude_prefixes=('_','.')
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath=root + os.sep +name  
            if filepath.endswith("NAO_sfc_all_events_mean.nc"):
                if not name.startswith(exclude_prefixes): 
                    r.append(os.path.join(root,name))
    return r

#obtain filepaths
filepaths=list_files("/path_to_file/")
filepaths.sort()

#labels for pandas DataFrame
models=['CCCma','HAMMOZ', 'HadGEM','INM','MIROC','MPI_HR', 'MPI_LR', 'NCAR', 'UKESM', 'WACCM']

#open reanlysis NAO file
DS2=xr.open_dataset('/path_to_file/file.nc')
ERA_NAO=DS2.msl.squeeze()
DS2.close()
#process data to calculate Euclidean distance
ERA_roll_NAO=ERA_NAO.rolling(time=5,center=True).mean()
ERA_roll_NAO=np.ma.masked_array(ERA_roll_NAO,np.isnan(ERA_roll_NAO))
ERA_mean=np.nanmean(ERA_roll_NAO)
ERA_std=np.nanstd(ERA_roll_NAO,ddof=1)
ERA_trans=(ERA_roll_NAO - ERA_mean)/ERA_std

#load CMIP, process timeseries, calculate Euclidean distance
distance=[]
for i in range(0, len(filepaths)):
    DS=xr.open_dataset(filepaths[i])
    NAO=DS.psl.squeeze()
    DS.close()
    roll=NAO.rolling(time=5,center=True).mean()
    roll=np.ma.masked_array(roll,np.isnan(roll))
    model_mean=np.nanmean(roll)
    model_std=np.nanstd(roll,ddof=1)
    model_trans=(roll - model_mean)/model_std
    SE=np.square(ERA_trans - model_trans)
    RMSE=math.sqrt(np.mean(SE))
    distance.append(RMSE)

distance_df=pd.DataFrame(distance,columns=['NAO'],index=models)
distance_df=distance_df.transpose()
print(distance_df)

#save DataFrame of Euclidean distances if required
distance_df.to_pickle('distance/rev_recalc/NAO_combined_distance_df.pkl')

