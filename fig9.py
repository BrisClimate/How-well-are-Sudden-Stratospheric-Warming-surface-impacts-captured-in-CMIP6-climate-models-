'''
Code to produce figure 9. Code takes as input the NAO values from models and ERA reanalysis, and produced the plot in figure 9.

'''

import numpy as np
import math
import os
import pickle
import pandas as pd
import xarray as xr 
import scipy.stats as st 
from matplotlib import pyplot as plt

#function to change time scale
def preproc(ds):
    ds=ds.assign_coords(time=range(-30,31))
    ds=ds.squeeze()
    ds=ds.transpose()
    return ds

#define function to find filepathsto CMIP data
CMIP_model_means=("NAO_sfc_all_events_mean_all_rev_recalc.nc")
CMIP_model_events=("NAO_sfc_all_events_all_rev_recalc.nc")

def list_files(dir,input):
    r=[]
    exclude_prefixes=('_','.')
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath=root + os.sep +name  
            if filepath.endswith(input):
                if not name.startswith(exclude_prefixes): 
                    r.append(os.path.join(root,name))
    return r

#filepaths to input each CMIPmodel mean
filepaths=list_files("path_to_dir",input= CMIP_model_means)
filepaths.sort()

#labels for legend, including correlations
models=['CanESM5 [0.64]','MPI_ESM-1-2-HAM [0.81]', 'HadGEM3-GC31-LL [0.83]','INM CM5-0 [0.80]','MIROC6 [0.79]','MPI-ESM1-2-HR [0.81]', 'MPI-ESM1-2-LR [0.81]', 'CESM2 [0.66]', 'UKESM1-0-LL [0.79]', 'CESM2 WACCM [0.74]']

#input ERA data mean
DS2=xr.open_dataset('path_to_file/file.nc')  
DS2=DS2.assign_coords(time=range(-30,31))
ERA_NAO=DS2.msl.squeeze()
DS2.close()

#calculate ERA CI
DS=xr.open_mfdataset('path_to_file/individual_SSW_file.nc',combine='nested',concat_dim='events',preprocess=preproc)
NAO_events=DS.msl
DS.close()

upper_CI=[]
lower_CI=[]

for t in range(0,len(NAO_events.time)):
	data=NAO_events[:,t]
	LCI,UCI=st.norm.interval(alpha=0.95,loc=np.mean(data),scale=st.sem(data))
	lower_CI.append(LCI)
	upper_CI.append(UCI)

# for Multimodel mean
event_filepaths=list_files("path_to_dir",input=CMIP_model_events)
event_filepaths.sort()
all_NAO_events=[]
for m in range(0,len(event_filepaths)):
    DS=xr.open_dataset(event_filepaths[m])
    events=DS.psl.squeeze()
    all_NAO_events.extend(events)
all_NAO_events=xr.concat((all_NAO_events),dim='events')
model_em=xr.DataArray.mean(all_NAO_events,dim='events')

#input model means
model_NAO=[]
for i in range(0, len(filepaths)):
    DS=xr.open_dataset(filepaths[i])
    NAO=DS.psl.squeeze()
    DS.close()
    model_NAO.extend(NAO)
    
model_NAO_arr=np.array(model_NAO).reshape(10,61)

combined_df=pd.DataFrame(model_NAO_arr, index=models, )
df2=pd.DataFrame(model_em,columns=['MMM']).transpose()
df3=pd.DataFrame(ERA_NAO,columns=['observed']).transpose()
combined_df=combined_df.append(df2)
combined_df=combined_df.append(df3)

colors=["#26538d","#750851","#247afd","#d767ad","#548d44","#69d84f","#a87900","#fd5956","#f9bc08","#840000"]

#input which models have significant correlations with reanalysis
combined_non_sigs=[]
combined_sigs=[0,1,2,3,4,5,6,7,8,9]

#plot figure
fig,ax=plt.subplots()
for i in combined_non_sigs:       
    ax.plot(ERA_NAO.time,model_NAO_arr[i],linewidth=0.75,color=colors[i])
for j in combined_sigs:          
    ax.plot(ERA_NAO.time,model_NAO_arr[j],linewidth=0.75,color=colors[j])
    ax.scatter(ERA_NAO.time,model_NAO_arr[j],s=4,zorder=3,color=colors[j])
    ax.plot([],[],'-o',color=colors[j],label=models[j])
ax.plot(ERA_NAO.time,ERA_NAO,color='black',zorder=3)
ax.plot(ERA_NAO.time,model_em, color='dimgray',zorder=2)
ax.plot(ERA_NAO.time,lower_CI,color='lightgray',zorder=1)             
ax.plot(ERA_NAO.time,upper_CI,color='lightgray',zorder=1)             
ax.fill_between(ERA_NAO.time,lower_CI,upper_CI,color='lightgray')    
ax.axhline(y=0,linestyle='--', color='black', linewidth=0.7)
ax.set_xlim(-30,30)
ax.set_ylim(-2.0,1.5)
ax.set_xlabel('lag /days relative to surface impact date',fontsize=13)
ax.set_ylabel('NAO Index $\sigma$',fontsize=13)
ax.set_title('NAO',loc='left',fontsize=13)      
plt.legend(loc='upper right',framealpha=0.6,ncol=3, fontsize=10)

fig1=plt.gcf()
plt.show()
fig1.savefig('NAO_combined_plus_corr.eps')
fig1.savefig('NAO_combined_plus_corr.pdf',format='pdf',dpi=500)
