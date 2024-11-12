import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mpl
from IOH_parser import IOH_Parser
from typing import List, Iterable
import os

##### -----------------------------------------------------------
##### ------------------CONSTANTS--------------------------------
##### -----------------------------------------------------------

BUDGET:int = 200
MAX_OFFICIAL_BUDGET:int = 1000
DIMENSIONS:list = [1,3,5,50]
RS_DIMENSIONS:list = [1,3,5]
INDEX_:int = 0

MAIN_PATH = os.path.dirname(os.path.realpath(__file__))

OBJECTIVE_NUMBER:int = 2
ITEM_NUMBER:int = 4

FORMAT:str = "png"
N_RUNS:int = 30

# Get some random number generator
NN = 7#np.random.randint(40)

OPTIMUM_COMBINATION:list = [-5, 
                            -5, 
                            -0.5199973575358707, 
                            3.9967929874454744, 
                        	-1.6569346182040572]

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
    
def mean_cumulative_objective_array_plot_dual(data_array_aniso:List[pd.DataFrame],
                                         data_array_iso:List[pd.DataFrame],
                                         ax_ptr:plt.Axes,
                                         data_array_RS:List[pd.DataFrame]=None,
                                         ):
    

    for idx1, obj in enumerate(data_array_aniso):
        if DIMENSIONS[idx1]==1:
            # In this setting continue as the values might not be comparable
            continue
        else:
            mean_arr,std_arr, count_arr  = (obj.groupby(['evaluations']).mean(), 
                                            obj.groupby(['evaluations']).std().fillna(0), 
                                            obj.groupby(['evaluations']).count())
            up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
            lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
            ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx1]}D - Anisotropic",color=f'C{idx1}')
            ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2,color=f'C{idx1}')
    
    for idx1, obj in enumerate(data_array_iso):
        mean_arr,std_arr, count_arr  = (obj.groupby(['evaluations']).mean(), 
                                        obj.groupby(['evaluations']).std().fillna(0), 
                                        obj.groupby(['evaluations']).count())
        up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
        lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
        ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{DIMENSIONS[idx1]}D - Isotropic",linestyle='dashed',color=f'C{idx1}')
        ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2,color=f'C{idx1}',linestyle='dashed')
        
    # Check if the Data Array for Random Search is not None
    if data_array_RS is not None:
        for idx1, obj in enumerate(data_array_RS):
            mean_arr,std_arr, count_arr  = (obj.groupby(['evaluations']).mean(), 
                                            obj.groupby(['evaluations']).std().fillna(0), 
                                            obj.groupby(['evaluations']).count())
            up_bound = -1*mean_arr['cum_min_objective'].to_numpy() + 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
            lo_bound = -1*mean_arr['cum_min_objective'].to_numpy() - 1.96*std_arr['cum_min_objective'].to_numpy()/np.sqrt(count_arr['cum_min_objective'].to_numpy())
            cur_rs_dim = RS_DIMENSIONS[idx1]
            global_idx = DIMENSIONS.index(cur_rs_dim)
            ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['cum_min_objective'].to_numpy(),label=f"{RS_DIMENSIONS[idx1]}D - RS",linestyle='dotted',color=f'C{global_idx}')
            
            ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  lo_bound, y2 = up_bound, alpha=0.2,color=f'C{global_idx}',linestyle='dotted')
    
    ax_ptr.set_xlabel("Evaluation Index")
    ax_ptr.set_ylabel("Objective")
    ax_ptr.legend(loc='best',
          ncol=2, fancybox=True,fontsize="11")

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

def plot_path_evolution(data_array:List[pd.DataFrame],ax_ptr:plt.Axes, fig_obj:mpl.Figure,
                        n_run:int=NN,analyzed_dims_idxs:Iterable=(1,5),
                        varX:str='x4',
                        varY:str='x0',
                        n_points:int = 30,
                        min_n_gen:int = 2):
    
    import re,math
    number1 = re.findall(r'\d+', varX)  # Finds all sequences of digits
    number1 = list(map(int, number1))    # Convert each number to integer
    
    number2 = re.findall(r'\d+', varY)  # Finds all sequences of digits
    number2 = list(map(int, number2))    # Convert each number to integer
    
    #lamda_calc = lambda N: int(4+ math.floor(3*np.log(N)))

    # This is set as constant markers
    markers_arr = ["s","+","p","o",'x']
    
    # Get the maximum dimension
    #max_dim = max([DIMENSIONS[aa] for aa in analyzed_dims_idxs])
    
    #total_evaluations = lamda_calc(max_dim)*min_n_gen
    
    
    
    
    for idx_,idx_1 in enumerate(analyzed_dims_idxs):
        # Get the runs to be analyzed
        #curDim:int = DIMENSIONS[idx_1]
        #cur_lamda:int = lamda_calc(curDim)
        #cur_mu:int = int(math.floor(cur_lamda)/2)
        #cur_generations:int =math.ceil(total_evaluations/cur_lamda)
        #cur_evaluations:int =cur_generations *cur_lamda
        
        # Extract the information about the current run
        analyzed_df:List[pd.DataFrame] = data_array[idx_1]
        sliced_df = analyzed_df[analyzed_df.loc[:,'run']==n_run]
        #parent_design = sliced_df.iloc[0:1,:]
        #mod_slice_df:pd.DataFrame = pd.concat([parent_design]*(cur_lamda-1),
        #                                      axis=0,ignore_index=True)
        #mod_slice_df:pd.DataFrame = pd.concat((mod_slice_df,sliced_df),
        #                                      axis=0,ignore_index=True)
        #start w
        low_b = sliced_df.index.min()
        #upper_b = low_b + cur_evaluations 
        upper_b = low_b + n_points + 1
        
        
        arr_xx = sliced_df.loc[low_b:upper_b,varX]
        arr_yy = sliced_df.loc[low_b:upper_b,varY]
        ax_ptr.scatter(arr_xx, arr_yy,s=6.4,
                    marker=markers_arr[idx_1],
                    color = f'C{idx_}',
                    label=f"{DIMENSIONS[idx_1]}D",)
        
        for idx, xy in enumerate(np.hstack((arr_xx.to_numpy().reshape((-1,1)),
                                               arr_yy.to_numpy().reshape((-1,1))))):
            ax_ptr.annotate(text=f"{idx+1}",xy=xy,ha="center",size=4.8)
        
        # Compute the mu per generation
        # for igen in range(cur_generations):
            
        #     analysis_frame:pd.DataFrame = mod_slice_df.iloc[0:cur_lamda*(igen+1),
        #                                                     :]
            
        #     analysis_frame_sorted:pd.DataFrame = analysis_frame.sort_values(by='Objective')
        #     cur_mu_star = analysis_frame_sorted.mean()
            
        #     if igen == 0:
        #         start_idx = low_b + 1
        #         end_idx = start_idx + cur_lamda

                
            
        #     else:
        #         start_idx = low_b + 1 + igen*cur_lamda
        #         end_idx = start_idx + cur_lamda
                
        #         #Plot the mu_star
        #         ax_ptr.scatter(cur_mu_star[varX],cur_mu_star[varY],
        #                     marker=markers_arr[idx_1],
        #                     color = f'C{idx_}')
                
        #         ax_ptr.annotate(text=f"$m_{igen}$",xy=[cur_mu_star[varX],
        #                                                cur_mu_star[varY]])
            
            
            
            
        #     cur_offspring:pd.DataFrame = sliced_df.loc[start_idx:end_idx-1,[varX,varY]]
        #     for idx,row in cur_offspring.iterrows():
        #         # Plot a line per each row
        #         ax_ptr.plot(np.vstack((cur_mu_star[varX],
        #                                row[varX])),
        #                     np.vstack((cur_mu_star[varY],
        #                                            row[varY])),
        #                     color = f'C{idx_}', alpha=0.1)
                    
                
                
            

    

    # Extract the best value on the inspected axis
    best_comb_1 = OPTIMUM_COMBINATION[number1[0]]
    best_comb_2 = OPTIMUM_COMBINATION[number2[0]]
    
    # Plot the best combination
    ax_ptr.scatter(best_comb_1, best_comb_2, marker='*',
                   color='magenta',label='Optimum')
    
    ax_ptr.set_xlabel(f"$X_{number1[0]}$")
    ax_ptr.set_ylabel(f"$X_{number2[0]}$")
    ax_ptr.set_title(f"Run {n_run}")

    ax_ptr.legend(loc="best")


        
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

# Change the path in this part of the code
os.chdir(mixed_path)

if OBJECTIVE_NUMBER == 1:
    if INDEX_!= 0:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_RS_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_RS-{eval}/IOHprofiler_f0_Star_Box_Problem_Middle.json")
    else:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual/IOHprofiler_f0_Star_Box_Problem_Middle.json")
        filep_RS_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_RS/IOHprofiler_f0_Star_Box_Problem_Middle.json")
else:
    if INDEX_!= 0:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual-{eval}/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_RS_root =lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_RS-{eval}/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
    else:
        filep_iso_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_iso_dual/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_root = lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_dual/IOHprofiler_f0_Star_Box_Problem_Controlled.json")
        filep_RS_root =lambda x,eval: os.path.join(mixed_path,f"Star_Box_CMA_ES_{x}D_RS/IOHprofiler_f0_Star_Box_Problem_Controlled.json")



file_roots = [filep_root(dim,INDEX_) for dim in DIMENSIONS]
file_iso_roots = [filep_iso_root(dim,INDEX_) for dim in DIMENSIONS]
file_RS_roots = [filep_RS_root(dim,INDEX_) for dim in RS_DIMENSIONS]

data_arr_RS = []
parser_list_RS:List[IOH_Parser] = []
if len(RS_DIMENSIONS) > 0:
    parser_list_RS:List[IOH_Parser] = [IOH_Parser(cur_fil) 
                                       for cur_fil in file_RS_roots]
    for idx, instance in enumerate(parser_list_RS):
        data_arr_RS.append(append_cumulative_minimum_to_Dataframe(instance.return_complete_table_per_instance(0)))


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
ax.legend(loc='best')

fig.savefig(f"Plot_raw_mean_average_normal_scale_iso.{FORMAT}",format=FORMAT,dpi=200)

# ----------------------------------------------------------------
# Plot 2
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)
cumulative_objective_array_plot(data_arr_iso,ax)
ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
ax.set_title("Cumulative Convergence Plots with Isotropic Covariance Matrix")
fig.savefig(f"Plot_all_cumulative_iso.{FORMAT}",format=FORMAT,dpi=200)

# Initialize a figure
fig, ax = plt.subplots(1,1)

mean_cumulative_objective_array_plot(data_arr_iso,ax)


ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=1,xmax=BUDGET)
ax.set_title("Cumulative Convergence Plot with Isotropic Covariance Matrix")
ax.legend(loc='best')

fig.savefig(f"Plot_mean_cumulative_normal_scale_iso.{FORMAT}",format=FORMAT,dpi=200)
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
ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
ax.legend(loc='best')

fig.savefig(f"Plot_mean_cumulative_log_scale_iso.{FORMAT}",format=FORMAT,dpi=200)




# Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr_iso,ax,fig)
    ax.set_xlim(xmax=200)
fig.savefig(f"parallel_coordinates_normal_scale_iso.{FORMAT}",format=FORMAT,dpi=200)

# # Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr_iso,ax,fig)
    ax.set_xscale('log')
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"parallel_coordinates_log_scale_iso.{FORMAT}",format=FORMAT,dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr_iso,ax,fig)
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"dual_objectives_normal_scale_iso.{FORMAT}",format=FORMAT,dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr_iso,ax,fig)
    ax.set_xscale('log')
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"dual_objectives_log_scale_iso.{FORMAT}",format=FORMAT,dpi=200)


for idx in range(N_RUNS):
    fig, axs = plt.subplots(1,1)
    plot_path_evolution(data_arr_iso,axs,fig,idx+1,[0,2,-1],varX='x4',
                        varY='x0',min_n_gen=1)
    #axs.set_xlim(xmin=-5,xmax=5)
    #axs.set_ylim(ymin=-5, ymax=5)
    axs.set_aspect('equal', adjustable='box')
    fig.savefig(f"path_evo_iso_run_{idx+1}_x4_v_x0.{FORMAT}",format="png",dpi=200)
    
    plt.close()
    fig, axs = plt.subplots(1,1)
    plot_path_evolution(data_arr_iso,axs,fig,idx+1,[0,2,-1],varX='x4',
                        varY='x1',min_n_gen=1)
    #axs.set_xlim(xmin=-5,xmax=5)
    #axs.set_ylim(ymin=-5, ymax=5)
    axs.set_aspect('equal', adjustable='box')
    fig.savefig(f"path_evo_iso_run_{idx+1}_x4_v_x1.{FORMAT}",format="png",dpi=200)
    plt.close()


# Initialize the parsing object
parser_list_2:List[IOH_Parser] = [IOH_Parser(cur_fil) for cur_fil in file_roots]
data_arr:List[pd.DataFrame] = []

for idx, instance in enumerate(parser_list_2):
    data_arr.append(append_cumulative_minimum_to_Dataframe(instance.return_complete_table_per_instance(0)))

plt.close()
# ----------------------------------------------------------------
# Plot 4
# ----------------------------------------------------------------




# Initialize a figure
fig, ax = plt.subplots(1,1)

# Run the function
mean_objective_array_plot(data_arr,ax)

ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=1,xmax=BUDGET)
ax.legend(loc='best')
ax.set_title("Mean Raw Data with Anisotropic Covariance Matrix")
fig.savefig(f"Plot_raw_mean_average_normal_scale.{FORMAT}",format=FORMAT,dpi=200)

# Initialize a figure
fig, ax = plt.subplots(1,1)
cumulative_objective_array_plot(data_arr,ax)
ax.set_title("Cumulative Convergence Plots with Anisotropic Covariance Matrix")
ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"Plot_all_cumulative.{FORMAT}",format=FORMAT,dpi=200)
# ----------------------------------------------------------------
# Plot 5
# ----------------------------------------------------------------

# Initialize a figure
fig, ax = plt.subplots(1,1)

mean_cumulative_objective_array_plot(data_arr,ax)
ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=1,xmax=BUDGET)
ax.legend(loc='best')
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
fig.savefig(f"Plot_mean_cumulative_normal_scale.{FORMAT}",format=FORMAT,dpi=200)


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
ax.legend(loc='best')
ax.set_title("Cumulative Convergence Plot with Anisotropic Covariance Matrix")
ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"Plot_mean_cumulative_log_scale.{FORMAT}",format=FORMAT,dpi=200)


# Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr,ax,fig)
    ax.set_xlim(xmin=1,xmax=BUDGET)
fig.savefig(f"parallel_coordinates_normal_scale.{FORMAT}",format=FORMAT,dpi=200)

# # Initialize a figure
fig, axs = plt.subplots(3,1,sharex='col')

for idx,ax in enumerate(axs):
    parallel_coordinates_plot(idx,data_arr,ax,fig)
    ax.set_xscale('log')
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"parallel_coordinates_log_scale.{FORMAT}",format=FORMAT,dpi=200)


# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr,ax,fig)
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)


fig.savefig(f"dual_objectives_normal_scale.{FORMAT}",format=FORMAT,dpi=200)

# Initialize a figure
fig, axs = plt.subplots(2,1,sharex='all')

for idx,ax in enumerate(axs):
    dual_objectives_plot(idx,data_arr,ax,fig)
    ax.set_xscale('log')
    ax.set_xlim(xmin=1,xmax=MAX_OFFICIAL_BUDGET)
fig.savefig(f"dual_objectives_log_scale.{FORMAT}",format=FORMAT,dpi=200)


for idx in range(N_RUNS):
    fig, axs = plt.subplots(1,1)
    plot_path_evolution(data_arr,axs,fig,idx+1,[0,2,-1],varX='x4',
                        varY='x0',min_n_gen=1)
    #axs.set_xlim(xmin=-5,xmax=5)
    #axs.set_ylim(ymin=-5, ymax=5)
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(f"path_evo_run_{idx+1}_x4_v_x0.{FORMAT}",format="png",dpi=200)
    
    plt.close()
    fig, axs = plt.subplots(1,1)
    plot_path_evolution(data_arr,axs,fig,idx+1,[0,2,-1],varX='x4',
                        varY='x1',min_n_gen=1)
    #axs.set_xlim(xmin=-5,xmax=5)
    #axs.set_ylim(ymin=-5, ymax=5)
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(f"path_evo_run_{idx+1}_x4_v_x1.{FORMAT}",format="png",dpi=200)
    plt.close()

fig, ax = plt.subplots(1,1)
mean_cumulative_objective_array_plot_dual(data_array_aniso=data_arr,
                                     data_array_iso=data_arr_iso,ax_ptr=ax,
                                     data_array_RS=None)
ax.set_xlim(xmin=1,xmax=BUDGET)
#ax.set_ylim(ymin=-1000, ymax=12500)
ax.set_title("Cumulative Convergence Plot - Combined")
fig.savefig(f"BOTH.{FORMAT}",format="png",dpi=200)
plt.close()

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
