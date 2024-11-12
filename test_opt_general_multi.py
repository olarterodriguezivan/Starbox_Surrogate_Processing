# Import the basic libraries
import os
import numpy as np
import csv
import pandas as pd
import multiprocessing as mp
import shutil
import time
from typing import List, Callable, Iterable

# Import PYCMA-ES (from Nikolaus Hansen)
import cma

# Import the Modular CMA-ES library (Jacob de Nobel, Diederick Vermetten)
from modcma import c_maes
from modcma import AskTellCMAES

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

BUDGET:int = 1000 # Manage a budget of simulations
DIMENSIONS:list = [1,3,5]
SIGMA_0:float = 2.5/2
N_RUNS:int =  50 # Number of runs
MAX_RESTARTS_DEFAULT:int = 10

# THIS IS A SWITCH TO CHANGE BETWEEN JUST ANALYZING EVERYTHING IN 5-D
# OR PERHAPS USING SUBDIMENSIONAL SURROGATES
DIMENSIONALITY_CASE:int = 1


# LIBRARY TO USE
# If set to '1' then use Niko Hansen's Library
# If '2', then use Modular CMA-ES library (Jacob de Nobel, Diederick Vermetten)
CMA_ES_HAND:int = 2

# Change this to define the case to evaluate the population size
POP_CASE = 2


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
        #funct_eval:float = self.__cur_sea - self.__lamda*np.maximum(self.__cur_intrusion-INTRUSION_PRIME,0)

        if self.meta_data.optimization_type.name == "MAX":
            return funct_eval
        else:
            return -1* funct_eval
    

            

    @property
    def regressor_model_sea(self)->GridSearchCV:
        return self.__regressor_model_sea
    
    @regressor_model_sea.deleter
    def regressor_model_sea(self):
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


def modify_array(x_0:np.ndarray):
    # This is the decorator factory function
    def decorator(func):
        def wrapper(x_inp:np.ndarray):
            # Apply modifications based on `modification_type`
            arr = np.array(x_inp).ravel()
            x_01 = np.array(x_0).ravel()
            if x_inp.size == 1:
                modified_array = np.array([x_01[0],x_01[1],x_01[2],x_01[3],arr[0]])
            elif x_inp.size == 3:
                modified_array = np.array([arr[0],arr[1],x_01[2],x_01[3],arr[2]])
            else:
                modified_array = np.array(x_inp).ravel()  # No modification if type is unknown

            # Call the original function with the modified array
            return func(modified_array)
        return wrapper
    return decorator


def return_simulation_setup(initial_val:np.ndarray,changing_dimensions:int=1,
                            initial_budget:int=BUDGET,full_sampler:bool=True)->cma.CMAOptions:
    
    if not initial_val.size  == 5:
        raise ValueError("The initial value should be of size 5")

    # Fill the fixed variables property
    if changing_dimensions == 1:
        fixed_vars = {
            0:initial_val[0,0],
            1:initial_val[0,1],
            2:initial_val[0,2],
            3:initial_val[0,3],
        }
    elif changing_dimensions ==3:
        fixed_vars = {
            2:initial_val[0,2],
            3:initial_val[0,3],
        }
    else:
        fixed_vars = {}

    # Initialize the CMA-ES Object
    # Options

    if full_sampler:
        samp:cma.sampler = cma.sampler.GaussFullSampler
    else:
        samp:cma.sampler = cma.sampler.GaussStandardConstant

    opts:cma.CMAOptions = cma.CMAOptions()
    opts.set({'bounds':[-5.0,5.0],
            'tolfun': 1e-12,
            'maxfevals':initial_budget,
            'CMA_sampler':samp,
            'fixed_variables':fixed_vars,
    })

    # Return the options of the CMA-ES algorithm

    return opts

def adjust_initial_input(x0:np.ndarray,dim:int):

    x_01:np.ndarray = x0.ravel()

    if not x0.size==5:
        raise ValueError("The size of the array must be 5")
    if dim ==1:
        return np.array([x_01[-1]]).ravel()
    elif dim ==3:
        return np.array([x_01[0], x_01[1],x_01[-1]]).ravel()
    elif dim ==5:
        return x_01
    else:
        raise ValueError("Dimension is badly set")


def return_initial_mu_lamda(dim:int, case:int)->List[int]:

    if dim not in (1,3,5):
        raise ValueError("The dimension should be an integer equal to 1,3 or 5")
    
    # Some lambda functions to compute the parent and offspring sizes
    default_lamda = lambda d: int(np.floor(4+3*np.log(d)))
    default_mu = lambda lamdda: int(np.ceil(lamdda/2))

    # Now evaluate the cases
    if case == 0:
        # The computation is let free depending on the dimension
        lamda = default_lamda(dim)
        return lamda,default_mu(lamda)
    
    elif case == 1:
        lamda = default_lamda(1)

    elif case == 2:
        lamda = default_lamda(3)

    elif case == 3:
        lamda = default_lamda(5)
    

    return default_mu(lamda), lamda




# Generate two properties from the IOH framework to account for 
# intrusion and specific energy absorption metrics


##### -----------------------------------------------------------
##### ------------------DEFINITION-------------------------------
##### -----------------------------------------------------------

# Change the working directory
os.chdir('/home/ivanolar/Documents')

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

    if CMA_ES_HAND ==1:
        extra_str = "Hansen"
    elif CMA_ES_HAND ==2:
        extra_str = "Vermetten-De_Nobel"
    logger_ioh = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"Star_Box_CMA_ES_{dim}D_dual",       # in a folder named: 'my-experiment'
        algorithm_name=f"CMA-ES_{extra_str}",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )


    logger_ioh_2 = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"Star_Box_CMA_ES_{dim}D_iso_dual",       # in a folder named: 'my-experiment'
        algorithm_name=f"CMA-ES_iso_{extra_str}",    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,
    )

    logger_ioh_cont_1.append(logger_ioh)
    logger_ioh_cont_2.append(logger_ioh_2)


# Have some new space...
del logger_ioh, logger_ioh_2

# Logger
#logger:cma.CMADataLogger= cma.CMADataLogger()
#logger_2:cma.CMADataLogger= cma.CMADataLogger()


bestever = cma.optimization_tools.BestSolution()
bestever_2 = cma.optimization_tools.BestSolution()

ff:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,5,1)
ff2:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,5,2)

for run in range(N_RUNS):

    x0 = np.random.uniform(-5,5,(1,5))

    
    
    for idx, dim in enumerate(DIMENSIONS):

        mu_0, lamda_0 = return_initial_mu_lamda(dim=dim,case=POP_CASE)
        
        # Update the dimensionality track for IOH
        ff.opt_dimensionality = dim
        ff2.opt_dimensionality = dim

        ff.lamda = LAMDA
        ff2.lamda = LAMDA


        # Set the tracks for the logger (Namely intrusion and sea)
        logger_ioh_cont_1[idx].watch(ff,['cur_intrusion','cur_sea'])
        logger_ioh_cont_1[idx].add_run_attributes(ff,['opt_dimensionality','lamda'])


        logger_ioh_cont_2[idx].watch(ff2,['cur_intrusion','cur_sea'])
        logger_ioh_cont_2[idx].add_run_attributes(ff2,['opt_dimensionality','lamda'])

        ff.attach_logger(logger_ioh_cont_1[idx])
        ff2.attach_logger(logger_ioh_cont_2[idx])


        if CMA_ES_HAND == 1:


            opts = return_simulation_setup(x0,dim,initial_budget=BUDGET,full_sampler=True)
            opts2 = return_simulation_setup(x0,dim,initial_budget=BUDGET,full_sampler=False)


            try:
                res_1:cma.evolution_strategy.CMAEvolutionStrategyResult = cma.fmin(ff,x0=x0.flatten().tolist(),sigma0=SIGMA_0,
                                                                                options=opts,
                                                                                restarts=MAX_RESTARTS_DEFAULT,eval_initial_x=True)
                res_2:cma.evolution_strategy.CMAEvolutionStrategyResult = cma.fmin(ff2,x0=x0.flatten().tolist(), sigma0=SIGMA_0,
                                                                                options=opts2,
                                                                                eval_initial_x=True,
                                                                                restarts=MAX_RESTARTS_DEFAULT)
            
            except ValueError as e:
                raise ValueError(e.args,"There was the same error")
            
            

        elif CMA_ES_HAND==2:


            # Evaluate the functions
            #ff(x0.ravel())
            #ff2(x0.ravel())

            @modify_array(x0)
            def obj_func_1(x_, func:Starbox_problem=ff) -> float:
                return func(x_)
            
            @modify_array(x0)
            def obj_func_2(x_, func:Starbox_problem=ff2) -> float:
                return func(x_)
            
            obj_func_1(x0)
            obj_func_2(x0)


            module_1 = c_maes.parameters.Modules()
            module_2 = c_maes.parameters.Modules()

            if MAX_RESTARTS_DEFAULT == 0:
                module_1.restart_strategy = c_maes.options.RestartStrategy.NONE
                module_2.restart_strategy = c_maes.options.RestartStrategy.NONE
            else:
                module_1.restart_strategy = c_maes.options.RestartStrategy.IPOP
                module_2.restart_strategy = c_maes.options.RestartStrategy.IPOP

            #module_1.restart_strategy = c_maes.options.RESTART
            #module_2.restart_strategy = c_maes.options.RESTART

            module_1.bound_correction = c_maes.options.CorrectionMethod.SATURATE
            module_2.bound_correction = c_maes.options.CorrectionMethod.SATURATE

            module_1.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
            module_2.matrix_adaptation = c_maes.options.MatrixAdaptationType.NONE

            x01 = adjust_initial_input(x0=x0,dim=dim)

            opts = c_maes.parameters.Settings(dim=dim, modules=module_1,x0=x01, 
                                              sigma0 =SIGMA_0, budget=BUDGET, verbose=True,
                                              mu0 = mu_0, lambda0 = lamda_0,
                                              lb=np.array([-5.0]*dim), 
                                              ub=np.array([5.0]*dim))
            
            opts2 = c_maes.parameters.Settings(dim=dim, modules=module_2,x0=x01, 
                                               sigma0 =SIGMA_0, budget=BUDGET, verbose=True,
                                               mu0 = mu_0, lambda0 = lamda_0,
                                               lb=np.array([-5.0]*dim), 
                                               ub=np.array([5.0]*dim))

            parameters = c_maes.Parameters(opts)
            parameters2 = c_maes.Parameters(opts2)

            cma1 = c_maes.ModularCMAES(parameters)
            cma2 = c_maes.ModularCMAES(parameters2)

            cma1.run(obj_func_1)
            #cma1.run(ff)
            #while not cma1.break_conditions():
            #    cma1.step(ff)
            cma2.run(obj_func_2)

        else:
            raise NotImplementedError("No other libraries besides hansen's and de nobel's are used")





        # Evolution Strategy Object
        #es:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
        #                                                    inopts=opts)
        
        #es2:cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=x0_1,sigma0=SIGMA_0,
        #                                                    inopts=opts2)

        #logger.register(es,append=bestever.evalsall)

        #logger_2.register(es2,append=bestever_2.evalsall)
    
        # # Run the optimization loop
        # # while not es.stop():
        # #     try:
        # #         solutions = np.array(es.ask()).reshape((-1,dim))
        # #     except ValueError as e:
        # #         print(e.args)

        # #     es.tell(solutions, [ff(x) for x in solutions])
        # #     es.logger.add()  # write data to disc to be plotted
        # #     es.disp()

        # try:
        #     es.optimize(ff,n_jobs=0)
        # except ValueError as e:
        #     raise ValueError(e.args)
        #     #es.optimize(ff,n_jobs=0)
        #     #es.optimize(ff)
        # except Exception as e:
        #     print("Something happened: ", e.args)
        # else:
        #     es.result_pretty()
        #     cma.s.pprint(es.best.__dict__)
        #     bestever.update(es.best)

        

        # # while not es2.stop():
            
        # #     try:
        # #         solutions2 = np.array(es2.ask()).reshape((-1,dim))
        # #     except ValueError as e:
        # #         print(e.args)
            
        # #     es2.tell(solutions2, [ff2(x) for x in solutions2])
        # #     es2.logger.add()  # write data to disc to be plotted
        # #     es2.disp()
        # try:
        #     es2.optimize(ff2, n_jobs=0)
        # except ValueError as e:
        #     raise ValueError(e.args)
        #     #es2.optimize(ff2, n_jobs=0)
        #     #es2.optimize(ff2)
        # except Exception as e:
        #     print("Something happened: ", e.args)
        # else:
        #     es2.result_pretty()
        #     #cma.s.pprint(es2.best.__dict__)
        #     bestever_2.update(es2.best)

        #logger.plot()
        #logger.save_to(f"CMA_ES_aniso/optim_dim_{dim}_")
        #logger_2.save_to(f"CMA_ES_iso/optim_dim_{dim}_")


        ff.reset()
        ff2.reset()

        ff.detach_logger()
        ff2.detach_logger()