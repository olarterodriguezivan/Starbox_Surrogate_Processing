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

# Search Folder to look for the reduced dataset
REDUCED_DATASET_NAME:str = "Reduced_Data"

# ---------------------------------------------
# --------------DEFINITION---------------------
# ---------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documents')

# Get the directories
data_dir_gen:str =  os.path.join(os.getcwd(),"data")

intrusion_dir = os.path.join(os.getcwd(),"data-Intrusion")
sea_dir = os.path.join(os.getcwd(),"data-SEA")

X_intrusion = np.loadtxt(os.path.join(intrusion_dir,REDUCED_DATASET_NAME,"x.csv"))
X_sea = np.loadtxt(os.path.join(sea_dir,REDUCED_DATASET_NAME,"x.csv"))

y_intrusion = np.loadtxt(os.path.join(intrusion_dir,REDUCED_DATASET_NAME,"intrusion_y.csv"))
y_sea = np.loadtxt(os.path.join(sea_dir,REDUCED_DATASET_NAME,"sea_y.csv"))


rf_reg_gen:RandomForestRegressor = RandomForestRegressor(verbose=False)

# Instatiate a Random Forest Regressor
rf_reg:RandomForestRegressor = RandomForestRegressor(n_estimators=10,criterion='friedman_mse',max_features="log2", verbose=True)

# Make a Grid Search
# params_1:dict = {'kernel':('rbf','linear','poly'),
#                  'C':np.logspace(start=-3, stop= 5,base=10, num=8),
#                 'epsilon': np.logspace(start=-6, stop= 1,base=10, num=8)     }
params_rf:dict = {'n_estimators':np.arange(5,300,5).tolist(),
                 'criterion':('squared_error','friedman_mse'),
                 'max_depth':np.arange(16,step=2).tolist()}


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

grid_s_rf_dual:GridSearchCV = GridSearchCV(rf_reg_gen,
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

grid_s_rf_dual.fit(X_sea,np.hstack((y_intrusion.reshape((-1,1)),
                                    y_sea.reshape((-1,1))
                                    )))

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
     file="RF_reg_intrusion_sub.skops",
     )

# Save the SEA estimator
dump(obj=grid_s_rf_sea,
     file="RF_reg_sea_sub.skops",
     )

# Save the dual estimator
dump(obj=grid_s_rf_dual,
     file="RF_reg_dual_sub.skops",
     )





