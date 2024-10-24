# Import the basic libraries
import os
import numpy as np
import csv
import pandas as pd
import multiprocessing as mp
import shutil
import time
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

# THIS IS TO SET THE LAMBDA TO ADJUST THE RESTRICTION ON THE PROBLEM
LAMDA:float = 3.981071705534969283e+02
INTRUSION_PRIME:float = 60.00

BUDGET:int = 1000 # Manage a budget of 
DIMENSIONS:list = [1,3,5]
SIGMA_0:float = (0.25/1)*10
N_RUNS:int =  50 # Number of runs
MAX_RESTARTS_DEFAULT:int = 5

##### -----------------------------------------------------------
##### ------------------HELPER FUNCTIONS-------------------------
##### -----------------------------------------------------------


class Starbox_problem(ioh.problem.RealSingleObjective):
    def __init__(self, regressor_model_intrusion:GridSearchCV,
                 regressor_model_sea:GridSearchCV, n_variables: int = 5, instance: int = 1,is_minimisation:bool = True,
                 opt_dimensionality:int=5, lamda:float = LAMDA):
        
        bounds = ioh.iohcpp.RealBounds(n_variables, -5, 5)
        optimum = ioh.iohcpp.RealSolution([0]* n_variables, 0.0)
        super().__init__('Star_Box_Problem', n_variables, instance, is_minimisation, bounds, [],  optimum)
  
        self.__regressor_model_intrusion:GridSearchCV = regressor_model_intrusion
        self.__regressor_model_sea:GridSearchCV = regressor_model_sea

        self.__cur_intrusion:float = -np.inf
        self.__cur_sea:float = -np.inf

        self.__opt_dimensionality:int = opt_dimensionality

        self.__lamda:float = lamda
 
    def evaluate(self, X:np.ndarray):
        
        # Reshape the array to be read by the regressor models
        X_ = X.reshape((1,-1))
        self.__cur_intrusion:float = self.__regressor_model_intrusion.predict(X=X_)[0]
        self.__cur_sea:float = self.__regressor_model_sea.predict(X=X_)[0]

        funct_eval:float = self.__cur_sea - self.__lamda*np.abs(self.__cur_intrusion-INTRUSION_PRIME)

        if self.meta_data.optimization_type.MAX:
            return funct_eval
        else:
            return -1* funct_eval

    
    @property
    def regressor_model_sea(self)->GridSearchCV:
        return self.__regressor_model_sea
    
    @regressor_model_sea.deleter
    def regressor_model(self):
        del self.__regressor_model_sea

    @property
    def regressor_model_intrusion(self)->GridSearchCV:
        return self.__regressor_model_intrusion
    
    @regressor_model_intrusion.deleter
    def regressor_model(self):
        del self.__regressor_model_intrusion
    
    @property
    def cur_intrusion(self)->float:
        return self.__cur_intrusion
    
    @property
    def cur_sea(self)->float:
        return self.__cur_sea
    
    @property
    def opt_dimensionality(self)->int:
        return self.__opt_dimensionality
    
    @opt_dimensionality.setter
    def opt_dimensionality(self,new_opt_dimensionality:int)-> None:
        self.__opt_dimensionality = new_opt_dimensionality
    
    @property
    def lamda(self)->float:
        return self.__lamda
    
    @lamda.setter
    def lamda(self,new_lamda:float)->None:
        self.__lamda = new_lamda
    

    

class Input_Manager:

    def __init__(self, X0:np.ndarray, regressor_model_intrusion:GridSearchCV,
                 regressor_model_sea:GridSearchCV,
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

        self.__regressor_model_intrusion:GridSearchCV = regressor_model_intrusion
        self.__regressor_model_sea:GridSearchCV = regressor_model_sea
        self.__minimisation:bool = minimisation

        # Get the properties of evaluation when calling the __call__ function
        self.__cur_eval = None
        self.__cur_sea = None
        self.__cur_intrusion = None

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

        self.__cur_eval = x_inp
        self.__cur_intrusion = self.regressor_model_intrusion.predict(X=x_inp)
        self.__cur_sea = self.regressor_model_sea.predict(X=x_inp)

        funct_eval:float = self.regressor_model_sea.predict(X=x_inp) - LAMDA*np.abs(self.regressor_model_intrusion.predict(X=x_inp)-INTRUSION_PRIME)
        if self.minimisation:
            return funct_eval
        else:
            return -1* funct_eval



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
    def regressor_model_sea(self)->GridSearchCV:
        return self.__regressor_model_sea
    
    @regressor_model_sea.deleter
    def regressor_model(self):
        del self.__regressor_model_sea

    @property
    def regressor_model_intrusion(self)->GridSearchCV:
        return self.__regressor_model_intrusion
    
    @regressor_model_intrusion.deleter
    def regressor_model(self):
        del self.__regressor_model_intrusion

    @property
    def cur_intrusion(self)->float:
        return self.__cur_intrusion
    
    @property
    def cur_sea(self)->float:
        return self.__cur_sea

# Generate two properties from the IOH framework to account for 
# intrusion and specific energy absorption metrics


##### -----------------------------------------------------------
##### ------------------DEFINITION-------------------------------
##### -----------------------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documentos')

# Load the model saved
# The trusted parameter can be adjusted;
# Check: https://www.kaggle.com/code/unofficialmerve/persisting-your-scikit-learn-model-using-skops
model_intrusion: GridSearchCV= load(file="RF_reg_intrusion.skops",trusted=True)
#print(model_intrusion.get_params(deep=True))

model_sea: GridSearchCV= load(file="RF_reg_sea.skops",trusted=True)
#print(model_sea.get_params(deep=True))

triggers:list = [
        ioh.logger.trigger.ALWAYS
    ]


logger_ioh_cont_1:List[ioh.logger.Analyzer] = []

logger_ioh_cont_2:List[ioh.logger.Analyzer] = []


for idx,dim in enumerate(DIMENSIONS):

    logger_ioh = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"Star_Box_CMA_ES_{dim}D_dual",       # in a folder named: 'my-experiment'
        algorithm_name="CMA-ES",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )


    logger_ioh_2 = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"Star_Box_CMA_ES_{dim}D_iso_dual",       # in a folder named: 'my-experiment'
        algorithm_name="CMA-ES_iso",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )

    logger_ioh_cont_1.append(logger_ioh)
    logger_ioh_cont_2.append(logger_ioh_2)


# Have some new space...
del logger_ioh, logger_ioh_2

# Logger
logger:cma.CMADataLogger= cma.CMADataLogger()
logger_2:cma.CMADataLogger= cma.CMADataLogger()


bestever = cma.optimization_tools.BestSolution()
bestever_2 = cma.optimization_tools.BestSolution()

ff:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,5,1)
ff2:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,5,2)

for run in range(N_RUNS):

    x0 = np.random.uniform(-5,5,(1,5))
    
    for idx, dim in enumerate(DIMENSIONS):
        
        x0_1 = x0.reshape((1,-1))
        if dim == 1:
     
            fixed_vars = {
                0:x0[0,0],
                1:x0[0,1],
                2:x0[0,2],
                3:x0[0,3],
            }
        elif dim ==3:
            fixed_vars = {
                2:x0[0,2],
                3:x0[0,3],
            }
        else:
            
            fixed_vars = {}
        
        # Start CMA ES
        # Options
        opts:cma.CMAOptions = cma.CMAOptions()
        opts.set({'bounds':[-5.0,5.0],
                'tolfun': 1e-12,
                'maxfevals':BUDGET,
                'CMA_sampler':cma.sampler.GaussFullSampler,
                'fixed_variables':fixed_vars
        })
                #'verb_append':bestever.evalsall})

        opts2:cma.CMAOptions = cma.CMAOptions()
        opts2.set({'bounds':[-5.0,5.0],
                'tolfun': 1e-12,
                'maxfevals':BUDGET,
                'CMA_sampler':cma.sampler.GaussStandardConstant,
                'fixed_variables':fixed_vars,
        })

        ff.opt_dimensionality = dim
        ff2.opt_dimensionality = dim

        logger_ioh_cont_1[idx].watch(ff,['cur_intrusion','cur_sea'])
        logger_ioh_cont_1[idx].add_run_attributes(ff,['opt_dimensionality','lamda'])


        logger_ioh_cont_2[idx].watch(ff2,['cur_intrusion','cur_sea'])
        logger_ioh_cont_2[idx].add_run_attributes(ff2,['opt_dimensionality','lamda'])
        

        ff.attach_logger(logger_ioh_cont_1[idx])
        ff2.attach_logger(logger_ioh_cont_2[idx])

        # Run the default initial value
        ff(x0_1)
        ff2(x0_1)


        # Evolution Strategy Object
        es:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
                                                            inopts=opts)
        
        es2:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
                                                            inopts=opts2)

        logger.register(es,append=bestever.evalsall)

        logger_2.register(es2,append=bestever_2.evalsall)
    
        # Run the optimization loop
        # while not es.stop():
        #     try:
        #         solutions = np.array(es.ask()).reshape((-1,dim))
        #     except ValueError as e:
        #         print(e.args)

        #     es.tell(solutions, [ff(x) for x in solutions])
        #     es.logger.add()  # write data to disc to be plotted
        #     es.disp()

        try:
            es.optimize(ff,n_jobs=0)
        except ValueError as e:
            print(e.args)
            #es.optimize(ff,n_jobs=0)
            #es.optimize(ff)
        except Exception as e:
            print("Something happened: ", e.args)
        else:
            es.result_pretty()
            cma.s.pprint(es.best.__dict__)
            bestever.update(es.best)

        

        # while not es2.stop():
            
        #     try:
        #         solutions2 = np.array(es2.ask()).reshape((-1,dim))
        #     except ValueError as e:
        #         print(e.args)
            
        #     es2.tell(solutions2, [ff2(x) for x in solutions2])
        #     es2.logger.add()  # write data to disc to be plotted
        #     es2.disp()
        try:
            es2.optimize(ff2, n_jobs=0)
        except ValueError as e:
            print(e.args)
            #es2.optimize(ff2, n_jobs=0)
            #es2.optimize(ff2)
        except Exception as e:
            print("Something happened: ", e.args)
        else:
            es2.result_pretty()
            #cma.s.pprint(es2.best.__dict__)
            bestever_2.update(es2.best)

        #logger.plot()
        logger.save_to(f"CMA_ES_aniso/optim_dim_{dim}_")
        logger_2.save_to(f"CMA_ES_iso/optim_dim_{dim}_")


        ff.reset()
        ff2.reset()

        ff.detach_logger()
        ff2.detach_logger()