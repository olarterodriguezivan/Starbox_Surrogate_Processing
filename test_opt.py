# Import the basic libraries
import os
import numpy as np
import csv
import pandas as pd
import multiprocessing as mp
import shutil
from typing import List, Callable, Iterable

# Import PYCMA-ES
import cma

# IOH
import ioh

# Specific Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# SKOPTS library to load objects from Scikit-Learn
from skops.card import Card
from skops.card import parse_modelcard
from skops.io import dump, load


##### -----------------------------------------------------------
##### ------------------CONSTANTS--------------------------------
##### -----------------------------------------------------------

BUDGET:int = 1000 # Manage a budget of 
DIMENSIONS:list = [1,3,5]
SIGMA_0:float = 0.25*10
RUNS:int = 10

##### -----------------------------------------------------------
##### ------------------HELPER FUNCTIONS-------------------------
##### -----------------------------------------------------------



class Input_Manager:

    def __init__(self, X0:np.ndarray, regressor_model:GridSearchCV,
                 minimisation:bool=True)->None:
        
        # Set the initial vector
        if not isinstance(X0,np.ndarray):
            raise ValueError("Not a good input")
        
        X0_copy:np.ndarray = X0.ravel().copy()

        if not X0_copy.size == 5:
            raise ValueError(" The input must be of 5 elements")
        
        if (np.any(X0_copy>5.0) or np.any(X0_copy<-5.0)):
            raise ValueError("The input vector is not within bounds.... \n [-5,5]")
        
        # Set the property
        self.__X0:np.ndarray = X0_copy.copy()

        self.__regressor_model:GridSearchCV = regressor_model
        self.__minimisation:bool = minimisation

    def __call__(self, inp:np.ndarray):
        '''
        Use the __call__ function to call the function
        '''

        if isinstance(inp,list):
            inp:np.ndarray = np.array(inp,dtype=float)
        elif isinstance(inp,float) or isinstance(inp,int):
            inp:np.ndarray = np.array([inp],dtype=float)

        # Identify the size is correct
        if not (inp.size == 1 or inp.size == 3 or inp.size == 5):
            raise ValueError("The sizes are not correct")
        else:
            inp_copy = inp.ravel().copy()
        
        # Now set the cases
        if inp.size == 1:
            return self.funct_out(np.array([self.XO[0],
                                            self.XO[1],
                                            self.XO[2],
                                            self.XO[3],
                                            inp_copy[0]]).reshape((1,-1)))
        elif inp.size ==3:
            return self.funct_out(np.array([inp_copy[0],
                                            inp_copy[1],
                                            self.XO[2],
                                            self.XO[3],
                                            inp_copy[2]]).reshape((1,-1)))
        else:
            return self.funct_out(inp_copy.reshape((1,-1)))


    
    def funct_out(self,x_inp)->float:
        if self.minimisation:
            return self.regressor_model.predict(X=x_inp)
        else:
            return -1* self.regressor_model.predict(X=x_inp)



    @property
    def XO(self)->np.ndarray:
        return self.__X0

    @XO.setter
    def XO(self,new_X0:np.ndarray)->None:
        self.__X0 = np.array(new_X0).ravel()

    @property
    def minimisation(self)->bool:
        return self.__minimisation
    
    @minimisation.setter
    def minimisation(self,new_min:bool)->None:
        self.__minimisation = new_min

    @property
    def regressor_model(self)->GridSearchCV:
        return self.__regressor_model
    
    @regressor_model.deleter
    def regressor_model(self):
        del self.__regressor_model



##### -----------------------------------------------------------
##### ------------------DEFINITION-------------------------------
##### -----------------------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documentos')

# Load the model saved
# The trusted parameter can be adjusted;
# Check: https://www.kaggle.com/code/unofficialmerve/persisting-your-scikit-learn-model-using-skops
model: GridSearchCV= load(file="RF_reg.skops",trusted=True)
print(model.get_params(deep=True))

triggers:list = [
        ioh.logger.trigger.ALWAYS
    ]

logger_ioh = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name="Star_Box_CMA_ES",       # in a folder named: 'my-experiment'
        algorithm_name="CMA-ES",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )

logger_ioh_2 = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name="Star_Box_CMA_ES_iso",       # in a folder named: 'my-experiment'
        algorithm_name="CMA-ES_iso",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )

# Logger
logger:cma.CMADataLogger= cma.CMADataLogger()

logger_2:cma.CMADataLogger= cma.CMADataLogger()

x0 = np.random.uniform(-5,5,(1,5))

ioh.wrap_problem(Input_Manager(x0,model,False),
                    name="StarBox",dimension=5,
                        optimization_type=ioh.OptimizationType.MIN,
                        lb=-5.0,
                        ub=5.0)

for idx, dim in enumerate(DIMENSIONS):

    ff:ioh.problem.RealSingleObjective = ioh.get_problem("StarBox",dimension=dim,
                                                         instance=idx+1)

    ff.attach_logger(logger_ioh)

    ff2:ioh.problem.RealSingleObjective = ioh.get_problem("StarBox",dimension=dim,
                                                         instance=idx+2)
    
    ff2.attach_logger(logger_ioh_2)

    # Start CMA ES
    # Options
    opts:cma.CMAOptions = cma.CMAOptions()
    opts.set({'bounds':[-5.0,5.0],
              'tolfun': 1e-12,
              'maxfevals':BUDGET})

    opts2:cma.CMAOptions = cma.CMAOptions()
    opts2.set({'bounds':[-5.0,5.0],
               'tolfun': 1e-12,
               'maxfevals':BUDGET,
               'CMA_sampler':cma.sampler.GaussStandardConstant})
    
    if dim == 1:
        x0_1 = np.array(x0[0,-1]).reshape((1,-1))
    elif dim ==3:
        x0_1 = np.array([x0[0,0],x0[0,1],x0[0,4]]).reshape((1,-1))
    else:
        x0_1 = x0.reshape((1,-1))

    # Run the default initial value
    ff(x0_1)
    ff2(x0_1)


    # Evolution Strategy Object
    es:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
                                                        options=opts)
    
    es2:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
                                                        options=opts2)

    logger.register(es)

    logger_2.register(es2)
   
    # Run the optimization loop
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [ff(x) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    es.result_pretty()

    while not es2.stop():
        solutions2 = es2.ask()
        es2.tell(solutions2, [ff2(x) for x in solutions2])
        es2.logger.add()  # write data to disc to be plotted
        es2.disp()

    es2.result_pretty()

    #logger.plot()
    logger.save_to(f"CMA_ES_aniso/optim_dim_{dim}_")
    logger_2.save_to(f"CMA_ES_iso/optim_dim_{dim}_")

    
    ff.reset()
    ff2.reset()