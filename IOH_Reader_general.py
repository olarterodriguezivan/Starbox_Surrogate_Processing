import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IOH_parser import IOH_Parser
from typing import List
import os

##### -----------------------------------------------------------
##### ------------------CONSTANTS--------------------------------
##### -----------------------------------------------------------

BUDGET:int = 200
DIMENSIONS:list = [1,3,5]


# ---------------------------------------------------------------
# Compute the cumulative value of the objective -----------------
# ---------------------------------------------------------------

def append_cumulative_minimum_to_Dataframe(ioh_dataframe:pd.DataFrame)->pd.DataFrame:

    new_series:pd.Series = pd.Series(np.array([]))
    for idx, cur_run in enumerate(pd.unique(ioh_dataframe['run'])):
        actual_series:pd.Series = ioh_dataframe.loc[ioh_dataframe['run']==cur_run,"Objective"]
        cum_min_partial:pd.Series = actual_series.cummin()
        new_series = pd.concat((new_series,cum_min_partial))
    
    ioh_dataframe.insert(len(ioh_dataframe.columns),
                         "cum_min_objective",
                         new_series, True)
    
    return ioh_dataframe


# Get a path
filep_iso:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES_iso_dual-1/IOHprofiler_f1121_StarBox.json"
filep:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES_dual-1/IOHprofiler_f1121_StarBox.json"

# Initialize the parsing object
parser_1 = IOH_Parser(filep_iso)
data_arr_iso:List[pd.DataFrame] = []

for idx, instance in enumerate(range(parser_1.number_of_instances)):
    data_arr_iso.append(append_cumulative_minimum_to_Dataframe(parser_1.return_complete_table_per_instance(idx)))


# ----------------------------------------------------------------
# Plot 1
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_1.number_of_instances):
    mean_arr,std_arr, count_arr = data_arr_iso[idx].groupby(['evaluations']).mean(), data_arr_iso[idx].groupby(['evaluations']).std(), data_arr_iso[idx].groupby(['evaluations']).count()
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['Objective'],label=f"{DIMENSIONS[idx]}D")
    up_bound = -1*mean_arr['Objective'].to_numpy() + 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
    lo_bound = -1*mean_arr['Objective'].to_numpy() - 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  up_bound, y2 = lo_bound, alpha=0.2)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.set_title("Mean Raw Data with Isotropic Covariance Matrix")
ax.legend()

plt.show()

# ----------------------------------------------------------------
# Plot 2
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_1.number_of_instances):
    mean_arr,std_arr, count_arr  = data_arr_iso[idx].groupby(['evaluations']).mean(), data_arr_iso[idx].groupby(['evaluations']).std().fillna(0), data_arr_iso[idx].groupby(['evaluations']).count()
    up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx]}D")
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.set_title("Cumulative Convergence Plot with Isotropic Covariance Matrix")
ax.legend()

plt.show()

# ----------------------------------------------------------------
# Plot 3
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_1.number_of_instances):
    mean_arr,std_arr, count_arr  = data_arr_iso[idx].groupby(['evaluations']).mean(), data_arr_iso[idx].groupby(['evaluations']).std().fillna(0), data_arr_iso[idx].groupby(['evaluations']).count()
    up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx]}D")
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title("Cumulative Convergence Plot with Isotropic Covariance Matrix")
ax.legend()

plt.show()

# Initialize the parsing object
parser_2 = IOH_Parser(filep)
data_arr:List[pd.DataFrame] = []

for idx, instance in enumerate(range(parser_2.number_of_instances)):
    data_arr.append(append_cumulative_minimum_to_Dataframe(parser_2.return_complete_table_per_instance(idx)))


# ----------------------------------------------------------------
# Plot 4
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_2.number_of_instances):
    mean_arr,std_arr,count_arr = data_arr[idx].groupby(['evaluations']).mean(), data_arr[idx].groupby(['evaluations']).std(), data_arr[idx].groupby(['evaluations']).count()
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['Objective'],label=f"{DIMENSIONS[idx]}D")
    up_bound = -1*mean_arr['Objective'].to_numpy() + 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
    lo_bound = -1*mean_arr['Objective'].to_numpy()- 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  up_bound, y2 = lo_bound, alpha=0.2)

ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()
ax.set_title("Mean Raw Data with Anisotropic Covariance Matrix")
plt.show()


# ----------------------------------------------------------------
# Plot 5
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_2.number_of_instances):
    mean_arr,std_arr, count_arr  = data_arr[idx].groupby(['evaluations']).mean(), data_arr[idx].groupby(['evaluations']).std().fillna(0), data_arr[idx].groupby(['evaluations']).count()
    up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx]}D")
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
plt.show()


# ----------------------------------------------------------------
# Plot 6
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

for idx in range(parser_2.number_of_instances):
    mean_arr,std_arr, count_arr  = data_arr[idx].groupby(['evaluations']).mean(), data_arr[idx].groupby(['evaluations']).std().fillna(0), data_arr[idx].groupby(['evaluations']).count()
    up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
    ax.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx]}D")
    ax.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
plt.show()


""" fig, ax = plt.subplots(1,1)
ax.plot(data['evaluations'].to_numpy(),-1*data['Objective'].ewm(span=4).mean(),label="1D")
ax.plot(data2['evaluations'].to_numpy(),-1*data2['Objective'].ewm(span=7).mean(),label="3D")
ax.plot(data3['evaluations'].to_numpy(),-1*data3['Objective'].ewm(span=8).mean(), label="5D")


fig, ax = plt.subplots(1,1)
ax.plot(data['evaluations'].to_numpy(),-1*data['Objective'].cummin(),label="1D")
ax.plot(data2['evaluations'].to_numpy(),-1*data2['Objective'].cummin(),label="3D")
ax.plot(data3['evaluations'].to_numpy(),-1*data3['Objective'].cummin(), label="5D")
ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()

plt.show()

parser = IOH_Parser(filep)
data:pd.DataFrame = parser.return_complete_table_per_instance(0)
data2 = parser.return_complete_table_per_instance(1)
data3 = parser.return_complete_table_per_instance(2)

fig, ax = plt.subplots(1,1)
ax.plot(data['evaluations'].to_numpy(),-1*data['Objective'].ewm(span=4).mean(),label="1D")
ax.plot(data2['evaluations'].to_numpy(),-1*data2['Objective'].ewm(span=7).mean(),label="3D")
ax.plot(data3['evaluations'].to_numpy(),-1*data3['Objective'].ewm(span=8).mean(), label="5D")
ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
#ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()

plt.show()

fig, ax = plt.subplots(1,1)
ax.plot(data['evaluations'].to_numpy(),-1*data['Objective'].cummin(),label="1D")
ax.plot(data2['evaluations'].to_numpy(),-1*data2['Objective'].cummin(),label="3D")
ax.plot(data3['evaluations'].to_numpy(),-1*data3['Objective'].cummin(), label="5D")
ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()

plt.show() """
