# Import the libraries
import os
import numpy as np
import csv
import pandas as pd
import multiprocessing as mp
import shutil
from typing import List

# Specific Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from skops.card import Card
from skops.card import parse_modelcard

from skops.io import dump, load
# --------------------------------------------
# --------------CONSTANTS --------------------
# --------------------------------------------

# Import the data
# List with all the filenames to search

FILNAMES:list = [
    "x_sobol.csv",
    "x_morris.csv",
    "x_lhs.csv",
    "y_sobol.csv",
    "y_morris.csv",
    "y_lhs.csv"
]

# ---------------------------------------------
# --------------DEFINITION---------------------
# ---------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documentos')

# Get the directories
data_dir_gen:str =  os.path.join(os.getcwd(),"data")

intrusion_dir = os.path.join(os.getcwd(),"data-Intrusion")
sea_dir = os.path.join(os.getcwd(),"data-SEA")

# Generate an array of Numpy arrays to read
parameter_data_files_intrusion:List[np.ndarray] = []
parameter_data_files_sea:List[np.ndarray] = []
for ii in range(0,3):
    curfil_1 = os.path.join(intrusion_dir,FILNAMES[ii])
    parameter_data_files_intrusion.append(np.loadtxt(curfil_1))

    curfil_1 = os.path.join(sea_dir,FILNAMES[ii])
    parameter_data_files_sea.append(np.loadtxt(curfil_1))


objective_data_files_intrusion:List[np.ndarray] = []
objective_data_files_sea:List[np.ndarray] = []
for ii in range(3,6):
    curfil_1 = os.path.join(intrusion_dir ,FILNAMES[ii])
    objective_data_files_intrusion.append(np.loadtxt(curfil_1))

    curfil_1 = os.path.join(sea_dir ,FILNAMES[ii])
    objective_data_files_sea.append(np.loadtxt(curfil_1))


del ii, curfil_1 

X_intrusion:np.ndarray = parameter_data_files_intrusion[0].copy()
X_sea:np.ndarray = parameter_data_files_sea[0].copy()

for idx, a in enumerate(parameter_data_files_intrusion[1:3]):
    X_intrusion = np.vstack((X_intrusion,a))

for idx, a in enumerate(parameter_data_files_sea[1:3]):
    X_sea= np.vstack((X_sea,a))

y_intrusion:np.ndarray = objective_data_files_intrusion[0]
y_sea:np.ndarray = objective_data_files_sea[0]

for idx, a in enumerate(objective_data_files_intrusion[1:3]):
    y_intrusion = np.hstack((y_intrusion,a))

for idx, a in enumerate(objective_data_files_sea[1:3]):
    y_sea = np.hstack((y_sea,a))

# Delete some variables in memory 
del objective_data_files_intrusion, parameter_data_files_intrusion, idx, a
del objective_data_files_sea, parameter_data_files_sea


rf_reg_gen:RandomForestRegressor = RandomForestRegressor(verbose=False)

# Instatiate a Random Forest Regressor
rf_reg:RandomForestRegressor = RandomForestRegressor(n_estimators=10,criterion='friedman_mse',max_features="log2", verbose=True)

# Make a Grid Search
# params_1:dict = {'kernel':('rbf','linear','poly'),
#                  'C':np.logspace(start=-3, stop= 5,base=10, num=8),
#                 'epsilon': np.logspace(start=-6, stop= 1,base=10, num=8)     }
params_rf:dict = {'n_estimators':np.arange(5,300,5).tolist(),
                 'criterion':('squared_error','friedman_mse'),
                 'max_features':('log2','sqrt')}


# grid_s_1:GridSearchCV = GridSearchCV(svr_reg_gen,
#                                      param_grid=params_1,
#                                      verbose=True)

grid_s_rf_intrusion:GridSearchCV = GridSearchCV(rf_reg_gen,
                                     param_grid=params_rf,
                                     verbose=False,
                                     refit=True,
                                     n_jobs=4)

grid_s_rf_sea:GridSearchCV = GridSearchCV(rf_reg_gen,
                                     param_grid=params_rf,
                                     verbose=False,
                                     refit=True,
                                     n_jobs=4)

# Fit the models
# grid_s_1.fit(X,y)
grid_s_rf_intrusion.fit(X_intrusion,y_intrusion)

print("Random Forest Estimator for intrusion",
        grid_s_rf_intrusion.best_estimator_,
      grid_s_rf_intrusion.best_score_,
      grid_s_rf_intrusion.best_index_,
      grid_s_rf_intrusion.best_params_,
      grid_s_rf_intrusion.cv_results_)

grid_s_rf_sea.fit(X_sea,y_sea)

print("Random Forest Estimator for SEA",
    grid_s_rf_sea.best_estimator_,
      grid_s_rf_sea.best_score_,
      grid_s_rf_sea.best_index_,
      grid_s_rf_sea.best_params_,
      grid_s_rf_sea.cv_results_)
      

# Now use the skops to persist the model obtained (Just the Random Forest Regressor)

# Set the card
# card:Card = Card(grid_s_rf_intrusion)
# card.metadata.license = "mit"
# card.save("README.md")

# Save the intrusion estimator
dump(obj=grid_s_rf_intrusion,
     file="RF_reg_intrusion.skops",
     )

# Save the SEA estimator
dump(obj=grid_s_rf_sea,
     file="RF_reg_sea.skops",
     )





