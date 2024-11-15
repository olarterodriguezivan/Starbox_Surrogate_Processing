import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mpl
from IOH_parser import IOH_Parser
from typing import List
import os

##### -----------------------------------------------------------
##### ------------------CONSTANTS--------------------------------
##### -----------------------------------------------------------

BUDGET:int = 300
DIMENSIONS:list = [1,3,5,10,20,40,50]
INDEX_:int = 0

MAIN_PATH = "/home/ivanolar/Documents/Analysis Folder 7/"

OBJECTIVE_NUMBER:int = 1
ITEM_NUMBER:int = 5

# Get some random number generator
NN = np.random.randint(40)

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


def mean_objective_array_plot(data_array:List[pd.DataFrame],ax_ptr:plt.Axes):
    
    for idx in range(len(data_array)):
        mean_arr,std_arr, count_arr = (data_array[idx].groupby(['evaluations']).mean(), 
                                       data_array[idx].groupby(['evaluations']).std(), 
                                       data_array[idx].groupby(['evaluations']).count())
        
        ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['Objective'],label=f"{DIMENSIONS[idx]}D")
        up_bound = -1*mean_arr['Objective'].to_numpy() + 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
        lo_bound = -1*mean_arr['Objective'].to_numpy() - 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
        ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  up_bound, y2 = lo_bound, alpha=0.2)


def cumulative_objective_array_plot(data_array:List[pd.DataFrame],ax_ptr:plt.Axes):
    

    for idx1, obj in enumerate(data_array):
        for idx2, cur_run in enumerate(pd.unique(obj['run'])):
            # Get the sliced dataframe
            sliced_one = obj[obj.loc[:,'run']==cur_run]

            if cur_run != obj['run'].max():
                ax_ptr.plot(sliced_one["evaluations"].to_numpy(),
                            -1*sliced_one["cum_min_objective"].to_numpy(),
                            color=f"C{idx1}")
            else:
                ax_ptr.plot(sliced_one["evaluations"].to_numpy(),
                            -1*sliced_one["cum_min_objective"].to_numpy(),
                            color=f"C{idx1}", label=f"{DIMENSIONS[idx1]}D")
    
    ax_ptr.set_xlabel("Evaluation Index")
    ax_ptr.set_ylabel("Objective")
    ax_ptr.legend(loc="best")

def mean_cumulative_objective_array_plot(data_array:List[pd.DataFrame],ax_ptr:plt.Axes):
    
    for idx in range(len(data_array)):
        mean_arr,std_arr, count_arr  = (data_array[idx].groupby(['evaluations']).mean(), 
                                        data_array[idx].groupby(['evaluations']).std().fillna(0), 
                                        data_array[idx].groupby(['evaluations']).count())
        up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
        lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
        ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx]}D")
        ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)


def parallel_coordinates_plot(idx:int,data_array:List[pd.DataFrame],ax_ptr:plt.Axes, fig_obj:mpl.Figure, 
                              full_handler:bool =True):
    
    # Get a new DataFrame with an array with a mean
    mean_arr,std_arr, count_arr  = (data_array[idx].groupby(['evaluations'],as_index=False).mean(), 
                                    data_array[idx].groupby(['evaluations'],as_index=False).std().fillna(0), 
                                    data_array[idx].groupby(['evaluations'],as_index=False).count())
    
    color_list:list = [f'C{idx}' for idx in range(5)]

    if full_handler:
        for ii in range(5):
            up_bound = 1*mean_arr[f'x{ii}'].to_numpy() + 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
            lo_bound = 1*mean_arr[f'x{ii}'].to_numpy() - 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
            ax_ptr.plot(mean_arr['evaluations'].to_numpy(),mean_arr[f'x{ii}'].to_numpy(),label=f'x{ii}')
            ax_ptr.fill_between(x=mean_arr['evaluations'].to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)
    else:
        if idx == 0:
            pos_labels = [4]
            for ii in range(1):
                lab:str = f'x{pos_labels[ii]}'
                up_bound = 1*mean_arr[f'x{ii}'].to_numpy() + 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                lo_bound = 1*mean_arr[f'x{ii}'].to_numpy() - 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                ax_ptr.plot(mean_arr['evaluations'].to_numpy(),mean_arr[f'x{ii}'].to_numpy(),label=lab, 
                            color = color_list[pos_labels[ii]])
                ax_ptr.fill_between(x=mean_arr['evaluations'].to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2,
                                    color = color_list[pos_labels[ii]])

        elif idx == 1:
            pos_labels = [0,1,4]
            for ii in range(3):
                lab:str = f'x{pos_labels[ii]}'
                up_bound = 1*mean_arr[f'x{ii}'].to_numpy() + 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                lo_bound = 1*mean_arr[f'x{ii}'].to_numpy() - 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                ax_ptr.plot(mean_arr['evaluations'].to_numpy(),mean_arr[f'x{ii}'].to_numpy(),label=lab,
                            color = color_list[pos_labels[ii]])
                ax_ptr.fill_between(x=mean_arr['evaluations'].to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2,
                                    color = color_list[pos_labels[ii]])

        else:
            for ii in range(5):
                up_bound = 1*mean_arr[f'x{ii}'].to_numpy() + 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                lo_bound = 1*mean_arr[f'x{ii}'].to_numpy() - 1.96*std_arr[f'x{ii}'].to_numpy()/np.sqrt(count_arr[f'x{ii}'].to_numpy())
                ax_ptr.plot(mean_arr['evaluations'].to_numpy(),mean_arr[f'x{ii}'].to_numpy(),label=f'x{ii}')
                ax_ptr.fill_between(x=mean_arr['evaluations'].to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)

            
    
    ax_ptr.set_title(f"{DIMENSIONS[idx]}D")
    ax_ptr.set_ylim(bottom=-5,top=5)
    #ax_ptr.set_xlim(right=200)

    ax.set_ylabel("Parameter Value")
    if idx == 2:
        handles, labels = ax_ptr.get_legend_handles_labels()
        fig_obj.legend(handles, labels, loc='upper left')
        ax.set_xlabel("Evaluation Index")

def dual_objectives_plot(idx,data_array:List[pd.DataFrame],ax_ptr:plt.Axes, fig_obj:mpl.Figure,**kwargs):
    
    if idx ==0:
        col_name = "cur_sea"
        title = "SEA"
    else:
        col_name = "cur_intrusion"
        title = "Intrusion"

    for id in range(len(data_array)):
        mean_arr,std_arr, count_arr  = (data_array[id].groupby(['evaluations']).mean(), 
                                        data_array[id].groupby(['evaluations']).std().fillna(0), 
                                        data_array[id].groupby(['evaluations']).count())
        up_bound = 1*mean_arr[col_name].to_numpy() + 1.96*std_arr[col_name].to_numpy()/np.sqrt(count_arr[col_name].to_numpy())
        lo_bound = 1*mean_arr[col_name].to_numpy() - 1.96*std_arr[col_name].to_numpy()/np.sqrt(count_arr[col_name].to_numpy())
        ax_ptr.plot(mean_arr.index.to_numpy(),1*mean_arr[col_name].to_numpy(),label=f"{DIMENSIONS[id]}D")
        ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2)
    
    ax_ptr.set_title(title)

    # Get the properties
    #ax_ptr.set_xlim(right=200)

    if idx == 1:
        handles, labels = ax_ptr.get_legend_handles_labels()
        fig_obj.legend(handles, labels, loc='upper left')
        ax.set_xlabel("Evaluation Index")


        
# Use the os.path.walk to list all the json files possible of the folder
for maindirectory,other_subs, files in os.walk(MAIN_PATH, followlinks=True):
    for maindir, subdir_,file_ in list(zip(maindirectory,other_subs,files)):
        print(os.path.join(maindir,subdir_,file_))

# Get a path

# if INDEX_!= 0:
#     filep_iso_root = lambda x,eval: f"/home/ivanolar/Documents/Interesting Results/Star_Box_CMA_ES_{x}D_iso_dual-{eval}/IOHprofiler_f0_Star_Box_Problem.json"
#     filep_root = lambda x,eval: f"/home/ivanolar/Documents/Interesting Results/Star_Box_CMA_ES_{x}D_dual-{eval}/IOHprofiler_f0_Star_Box_Problem.json"
# else:
#     filep_iso_root = lambda x,eval: f"/home/ivanolar/Documents/Interesting Results/Star_Box_CMA_ES_{x}D_iso_dual/IOHprofiler_f0_Star_Box_Problem.json"
#     filep_root = lambda x,eval: f"/home/ivanolar/Documents/Interesting Results/Star_Box_CMA_ES_{x}D_dual/IOHprofiler_f0_Star_Box_Problem.json"


mixed_path = os.path.join(MAIN_PATH,f"Objective {OBJECTIVE_NUMBER}",f"{ITEM_NUMBER}")

if OBJECTIVE_NUMBER == 1:
    if INDEX_!= 0:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Middle.json")
    else:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual/IOHprofiler_f0_Star_Box_Problem_Middle.json")
else:
    if INDEX_!= 0:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
    else:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual/IOHprofiler_f0_Star_Box_Problem_Controlled.json")



file_roots = [filep_root(dim,INDEX_) for dim in DIMENSIONS]
file_iso_roots = [filep_iso_root(dim,INDEX_) for dim in DIMENSIONS]

# Initialize the parsing object
parser_list_1:List[IOH_Parser] = [IOH_Parser(cur_fil) for cur_fil in file_iso_roots]
data_arr_iso:List[pd.DataFrame] = []

for idx, instance in enumerate(parser_list_1):
    data_arr_iso.append(append_cumulative_minimum_to_Dataframe(instance.return_complete_table_per_instance(0)))


# ----------------------------------------------------------------
# Plot 1
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

# Run the function
mean_objective_array_plot(data_arr_iso,ax)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.set_title("Mean Raw Data with Isotropic Covariance Matrix")
ax.legend()

fig.savefig("Plot_raw_mean_average_normal_scale_iso",format="png",dpi=200)

# ----------------------------------------------------------------
# Plot 2
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)
cumulative_objective_array_plot(data_arr_iso,ax)

ax.set_title("Cumulative Convergence Plots with Isotropic Covariance Matrix")
fig.savefig("Plot_all_cumulative_iso",format="png",dpi=200)

# Initialize a figure
fig, ax = plt.subplots(1,1)

mean_cumulative_objective_array_plot(data_arr_iso,ax)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.set_title("Cumulative Convergence Plot with Isotropic Covariance Matrix")
ax.legend()


fig.savefig("Plot_mean_cumulative_normal_scale_iso",format="png",dpi=200)
# ----------------------------------------------------------------
# Plot 3
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

mean_cumulative_objective_array_plot(data_arr_iso,ax)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title("Cumulative Convergence Plot with Isotropic Covariance Matrix")
ax.legend()

fig.savefig("Plot_mean_cumulative_log_scale_iso",format="png",dpi=200)




# Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr_iso,ax,fig)
    ax.set_xlim(xmax=200)
fig.savefig("parallel_coordinates_normal_scale_iso",format="png",dpi=200)

# # Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr_iso,ax,fig)
    ax.set_xscale('log')
fig.savefig("parallel_coordinates_log_scale_iso",format="png",dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr_iso,ax,fig)

fig.savefig("dual_objectives_normal_scale_iso",format="png",dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr_iso,ax,fig)
    ax.set_xscale('log')
fig.savefig("dual_objectives_log_scale_iso",format="png",dpi=200)




# Initialize the parsing object
parser_list_2:List[IOH_Parser] = [IOH_Parser(cur_fil) for cur_fil in file_roots]
data_arr:List[pd.DataFrame] = []

for idx, instance in enumerate(parser_list_2):
    data_arr.append(append_cumulative_minimum_to_Dataframe(instance.return_complete_table_per_instance(0)))


# ----------------------------------------------------------------
# Plot 4
# ----------------------------------------------------------------




# Initialize a figure
fig, ax = plt.subplots(1,1)

# Run the function
mean_objective_array_plot(data_arr,ax)

ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()
ax.set_title("Mean Raw Data with Anisotropic Covariance Matrix")
fig.savefig("Plot_raw_mean_average_normal_scale",format="png",dpi=200)

# Initialize a figure
fig, ax = plt.subplots(1,1)
cumulative_objective_array_plot(data_arr,ax)
ax.set_title("Cumulative Convergence Plots with Anisotropic Covariance Matrix")
fig.savefig("Plot_all_cumulative",format="png",dpi=200)
# ----------------------------------------------------------------
# Plot 5
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

mean_cumulative_objective_array_plot(data_arr,ax)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
ax.legend()
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
fig.savefig("Plot_mean_cumulative_normal_scale",format="png",dpi=200)


# ----------------------------------------------------------------
# Plot 6
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)


mean_cumulative_objective_array_plot(data_arr,ax)

ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
fig.savefig("Plot_mean_cumulative_log_scale",format="png",dpi=200)


# Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr,ax,fig)
    ax.set_xlim(xmax=200)
fig.savefig("parallel_coordinates_normal_scale",format="png",dpi=200)

# # Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr,ax,fig)
    ax.set_xscale('log')
fig.savefig("parallel_coordinates_log_scale",format="png",dpi=200)


# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr,ax,fig)


fig.savefig("dual_objectives_normal_scale",format="png",dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr,ax,fig)
    ax.set_xscale('log')
fig.savefig("dual_objectives_log_scale",format="png",dpi=200)


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
