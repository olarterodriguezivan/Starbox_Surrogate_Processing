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

SELECTED_NUMBER:int = 17
# ---------------------------------------------
# --------------DEFINITION---------------------
# ---------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documentos')

# Get the directories
data_dir_gen:str =  os.path.join(os.getcwd(),"data")
data_dir_specific:str = os.path.join(os.getcwd(),f"data_{SELECTED_NUMBER}")

# Generate an array of Numpy arrays to read
parameter_data_files:List[np.ndarray] = []
for ii in range(0,3):
    curfil_1 = os.path.join(data_dir_specific ,FILNAMES[ii])
    parameter_data_files.append(np.loadtxt(curfil_1))


objective_data_files:List[np.ndarray] = []
for ii in range(3,6):
    curfil_1 = os.path.join(data_dir_specific ,FILNAMES[ii])
    objective_data_files.append(np.loadtxt(curfil_1))


del ii, curfil_1 

X:np.ndarray = parameter_data_files[0].copy()

for idx, a in enumerate(parameter_data_files[1:3]):
    X = np.vstack((X,a))

y:np.ndarray = objective_data_files[0]

for idx, a in enumerate(objective_data_files[1:3]):
    y = np.hstack((y,a))

# Delete some variables in memory 
del objective_data_files, parameter_data_files, idx, a

# Generate a train_test_split (sklearn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     random_state=1,
                                                     shuffle=True)



svr_reg_gen:SVR = SVR(verbose=True,gamma='auto')
# Instatiate a Support Vector Regressor
svr_reg:SVR = SVR(kernel='rbf',degree=3,gamma='auto',C=10000,epsilon=0.00001,verbose=True)

rf_reg_gen:RandomForestRegressor = RandomForestRegressor(verbose=False)
# Instatiate a Random Forest Regressor
rf_reg:RandomForestRegressor = RandomForestRegressor(n_estimators=10,criterion='friedman_mse',max_features="log2", verbose=True)

# Make a Grid Search
params_1:dict = {'kernel':('rbf','linear','poly'),
                 'C':np.logspace(start=-3, stop= 5,base=10, num=8),
                'epsilon': np.logspace(start=-6, stop= 1,base=10, num=8)     }
params_2:dict = {'n_estimators':np.arange(5,300,5).tolist(),
                 'criterion':('squared_error','friedman_mse'),
                 'max_features':('log2','sqrt')}


grid_s_1:GridSearchCV = GridSearchCV(svr_reg_gen,
                                     param_grid=params_1,
                                     verbose=True)

grid_s_2:GridSearchCV = GridSearchCV(rf_reg_gen,
                                     param_grid=params_2,
                                     verbose=False,
                                     refit=True,
                                     n_jobs=4)

# Fit the models
# grid_s_1.fit(X,y)
grid_s_2.fit(X,y)

print(grid_s_2.best_estimator_,
      grid_s_2.best_score_,
      grid_s_2.best_index_,
      grid_s_2.best_params_,
      grid_s_2.cv_results_)
      

# Now use the skops to persist the model obtained (Just the Random Forest Regressor)

# Set the card
card:Card = Card(grid_s_2)
card.metadata.license = "mit"
card.save("README.md")

dump(obj=grid_s_2,
     file="RF_reg.skops",
     )





