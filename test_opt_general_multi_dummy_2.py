# Import the basic libraries
import os
import numpy as np
import csv
import pandas as pd
import multiprocessing as mp
import shutil
import time
from typing import List, Callable, Iterable
import warnings
import sys

# Import PYCMA-ES (from Nikolaus Hansen)
import cma

# Import the Modular CMA-ES library (Jacob de Nobel, Diederick Vermetten)
from modcma import c_maes, ModularCMAES, Parameters


# IOH
import ioh

# Specific Scikit Learn Libraries
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#from sklearn.svm import SVR # Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline

# SKOPTS library to load objects from Scikit-Learn
#from skops.card import Card
#from skops.card import parse_modelcard
from skops.io import load #, dump,  get_untrusted_types


##### -----------------------------------------------------------
##### ------------------CONSTANTS--------------------------------
##### -----------------------------------------------------------

# FILE MANAGEMENT
SURROGATE_PATH:str = "Surrogate_Folder"
RESULTS_PATH:str = "Results"


# THIS IS TO SET THE LAMBDA TO ADJUST THE RESTRICTION ON THE PROBLEM
#LAMDA:float = 3.981071705534969283e+02
LAMDA:float = 9.05e+03
INTRUSION_PRIME:float = 60.00

BUDGET:int = 1000 # Manage a budget of simulations
DIMENSIONS:list = [1,3,5]
SIGMA_0:float = 2.5/2
N_RUNS:int =  10 # Number of runs
MAX_RESTARTS_DEFAULT:int = 10

# The kind of objective to be evaluated
DEFAULT_OBJECTIVE_TYPE:int = 2

# LIBRARY TO USE
# If set to '1' then use Niko Hansen's Library
# If '2', then use Modular CMA-ES library (Jacob de Nobel, Diederick Vermetten)
CMA_ES_HAND:int = 2

# Change this to define the case to evaluate the population size
POP_CASE:int = 0

# NOT CHANGING (Seed)
SEED:int = 42

# LOAD REGRESSORS MODE
# Set to 1 to use full surrogate or 2 to just subspace surrogate
DIMENSIONALITY_CASE:int = 1

##### -----------------------------------------------------------
##### -----------------------CONFIG------------------------------
##### -----------------------------------------------------------

os.chdir(os.path.dirname(os.path.abspath(__file__)))

##### -----------------------------------------------------------
##### ------------------HELPER CLASSES---------------------------
##### -----------------------------------------------------------


class Starbox_problem(ioh.problem.RealSingleObjective):
    def __init__(self, regressor_model_intrusion:GridSearchCV,
                 regressor_model_sea:GridSearchCV, 
                 n_variables: int = 5, instance: int = 1,is_minimisation:bool = True,
                 opt_dimensionality:int=5, lamda:float = LAMDA, intrusion_ref:float = INTRUSION_PRIME,
                 objective_type:int = 1):
        
        """
        This is the constructor method of this class;
        The constructor receives the following inputs:

        ---------------------------
        Inputs:
        - regressor_model_intrusion: An object which acts as a surrogate to predict the intrusion
        """
        
        bounds = ioh.iohcpp.RealBounds(n_variables, -5, 5)
        optimum = ioh.iohcpp.RealSolution([0]* n_variables, 0.0)

        self.__objective_type:int = objective_type

        if self.__objective_type == 1:
            prob_name = "Star_Box_Problem_Middle"
        elif self.__objective_type == 2:
            prob_name = "Star_Box_Problem_Controlled"

        super().__init__(prob_name, n_variables, instance, is_minimisation, bounds, [],  optimum)


        if isinstance(regressor_model_intrusion, GridSearchCV):
            self.__regressor_model_intrusion:GridSearchCV = regressor_model_intrusion
        else:
            raise ValueError(f"The regressor model should be of type Grid Search")

        if isinstance(regressor_model_sea, GridSearchCV):
            self.__regressor_model_sea:GridSearchCV = regressor_model_sea
        else:
            self.__regressor_model_sea:None = None

        self.__regressor_model_sea:GridSearchCV = regressor_model_sea

        self.__cur_intrusion:float = -np.inf
        self.__cur_sea:float = -np.inf

        self.__opt_dimensionality:int = opt_dimensionality

        self.__lamda:float = lamda

        self.__intrusion_ref:float = intrusion_ref
        

 
    def evaluate(self, X:np.ndarray):
        
        # Reshape the array to be read by the regressor models

        X_ = X.reshape((1,-1))
        # Compute the intrusion and SEA
        self._function_computation(X_=X_)

        if self.__objective_type == 1:
            funct_eval:float = self.__cur_sea - self.__lamda*np.abs(self.__cur_intrusion-self.__intrusion_ref)
        elif self.__objective_type == 2:
            funct_eval:float = self.__cur_sea - self.__lamda*np.maximum(self.__cur_intrusion-self.__intrusion_ref,0)
        #funct_eval:float = self.__cur_sea - self.__lamda*np.maximum(self.__cur_intrusion-INTRUSION_PRIME,0)

        if self.meta_data.optimization_type.name == "MAX":
            return funct_eval
        else:
            return -1* funct_eval
    

    def _function_computation(self,X_:np.ndarray)->None:
        if X_.size > 5:
            X_flat = X_.flatten()
            X_mod = np.array([X_flat[0],X_flat[1],X_flat[2],X_flat[3],X_flat[4]]).reshape((1,-1))
        else:
            X_mod = np.copy(X_)
        if not isinstance(self.__regressor_model_sea,GridSearchCV):

            arr = np.array(self.__regressor_model_intrusion.predict(X=X_mod)).ravel()

            self.__cur_intrusion = arr[0]
            self.__cur_sea = arr[1]
        
        else:
            self.__cur_intrusion:float = self.__regressor_model_intrusion.predict(X=X_mod)[0]
            self.__cur_sea:float = self.__regressor_model_sea.predict(X=X_mod)[0]



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



class RandomSearch:
    def __init__(self, budget:int, dim:int):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget:int = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        # A dynamic parameter updated by the algorithm
        self.a_tracked_parameter = None

        self.dim:int = dim

    def __call__(self, func:Callable):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            # This is hard-coded (but known)
            x = np.random.uniform(-5, 5,size=(1,self.dim))

            # Update the tracked parameter
            self.a_tracked_parameter = i ** 10

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)

# ========================================================================================================
# HELPER FUNCTIONS
# ========================================================================================================

def modify_array(x_0:np.ndarray):
    # This is the decorator factory function
    def decorator(func):
        def wrapper(x_inp:np.ndarray):
            # Apply modifications based on `modification_type`
            arr = np.array(x_inp).ravel()
            x_01 = np.array(x_0).ravel()
            if x_inp.size == 1:
                modified_array = x_01.copy()
                modified_array[4] = arr[0]
                modified_array = modified_array[0:5] 
            elif x_inp.size == 3:
                modified_array = x_01.copy()
                modified_array[0] = arr[0]
                modified_array[1] = arr[1]
                modified_array[4] = arr[2]
                modified_array = modified_array[0:5] 
            else:
                modified_array = arr.copy()  
                #for idx,elem in enumerate(arr):
                #    modified_array[idx] = elem
                #modified_array = modified_array[0:arr.size] 

            # Call the original function with the modified array
            return func(modified_array)
        return wrapper
    return decorator


def return_simulation_setup(initial_val:np.ndarray,seed:int= SEED,changing_dimensions:int=1,
                            initial_budget:int=BUDGET,full_sampler:bool=True)->cma.CMAOptions:

    initial_val = initial_val.ravel()

    # All fixed dimensions
    all_fixed_dims = {x: initial_val[x] for x in range(initial_val.size)}
    # Fill the fixed variables property
    if changing_dimensions == 1:
        all_fixed_dims.pop(4)
    elif changing_dimensions ==3:
        all_fixed_dims.pop(0)
        all_fixed_dims.pop(1)
        all_fixed_dims.pop(4)
    elif changing_dimensions >=5:
        for aa in range(changing_dimensions): all_fixed_dims.pop(aa)

    # Initialize the CMA-ES Object
    # Options
    fixed_vars = all_fixed_dims.copy()

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
            'seed':seed,
    })

    # Return the options of the CMA-ES algorithm

    return opts

def return_simulation_setup_2(n_restarts:int=0,
                              full_sampler:bool=True)->c_maes.ModularCMAES:
    """
    This is a function which acts as a "pseudo-overload" of previous function as the setup is 
    based to call the De Nobel-Vermetten implementation of the CMA-ES algorithm
    
    """

    # Initialize a module from the C++ based back-end
    module:c_maes.ModularCMAES= c_maes.parameters.Modules()

    if n_restarts == 0:
        module.restart_strategy = c_maes.options.RestartStrategy.NONE
    else:
        module.restart_strategy = c_maes.options.RestartStrategy.IPOP

    #module_1.restart_strategy = c_maes.options.RESTART
    #module_2.restart_strategy = c_maes.options.RESTART

    module.bound_correction = c_maes.options.CorrectionMethod.SATURATE

    if full_sampler:
        module.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
    else:
        module.matrix_adaptation = c_maes.options.MatrixAdaptationType.NONE

    # Return the options of the CMA-ES algorithm
    return module

def adjust_initial_input(x0:np.ndarray,dim:int):

    x_01:np.ndarray = x0.flatten()

    if dim ==1:
        return np.array([x_01[4]]).flatten()
    elif dim ==3:
        return np.array([x_01[0], x_01[1],x_01[4]]).flatten()
    elif dim >=5:
        return x_01[0:dim]
    else:
        raise ValueError("Dimension is badly set")


def return_initial_mu_lamda(dim:int, case_:int)->List[int]:

    if dim not in (1,3,5) and case_ !=0:
        warnings.warn("Using now case 0")
        case_=0
    
    # Some lambda functions to compute the parent and offspring sizes
    default_lamda = lambda d: 4 + int(np.floor(3*np.log(d)))
    default_mu = lambda lamdda: int(np.floor(lamdda/2))

    # Now evaluate the cases
    if case_ == 0:
        # The computation is let free depending on the dimension
        lamda = default_lamda(dim)
        return default_mu(lamda), lamda
    
    elif case_ == 1:
        lamda = default_lamda(1)

    elif case_ == 2:
        lamda = default_lamda(3)

    elif case_ == 3:
        lamda = default_lamda(5)
    

    return default_mu(lamda), lamda




# Generate two properties from the IOH framework to account for 
# intrusion and specific energy absorption metrics


##### -----------------------------------------------------------
##### ------------------DEFINITION-------------------------------
##### -----------------------------------------------------------

def main(pid_id,
        dim_case:int = DIMENSIONALITY_CASE,
        cma_case_handler:int = CMA_ES_HAND,
        budget:int=BUDGET,
        verbose:bool=False,
        lamda_val:float = LAMDA,
        obj_type:int = DEFAULT_OBJECTIVE_TYPE,
        pop_case:int = POP_CASE,
        max_restarts:int = MAX_RESTARTS_DEFAULT,
        n_runs:int = N_RUNS,
        sigma_0:float = SIGMA_0,
        random_seed:int = SEED,
        dimensions:List[int]= DIMENSIONS,
        x0_:np.ndarray = np.empty(shape=(1,5)),
        intrusion:float = INTRUSION_PRIME,
        append_random_search:bool = False,
        *args, **kwargs):

    random_search_dimensions:Iterable = None
    for key, item in kwargs.items():
        if key=="random_search_dimensions":
            random_search_dimensions:Iterable = item # Assign this as reference 
    
    
    # Change the working directory
    # os.chdir('/home/ivanolar/Documents')
    print("Working from:", os.getcwd(), flush=True)
    # Load the model saved
    # The trusted parameter can be adjusted;
    # Check: https://www.kaggle.com/code/unofficialmerve/persisting-your-scikit-learn-model-using-skops


    if dim_case == 1:
        model_intrusion: GridSearchCV= load(file=os.path.join(os.getcwd(),
                                                              SURROGATE_PATH,"RF_reg_intrusion.skops"),
                                                              trusted=True)
        #print(model_intrusion.get_params(deep=True))

        model_sea: GridSearchCV= load(file=os.path.join(os.getcwd(),
                                                        SURROGATE_PATH,"RF_reg_sea.skops"),trusted=True)
        #print(model_sea.get_params(deep=True))



        model_dual: GridSearchCV= load(file=os.path.join(os.getcwd(),
                                                         SURROGATE_PATH,"RF_reg_dual_5D.skops"),trusted=True)

        
    else: 
        models_intrusion:List[GridSearchCV] = [load(os.path.join(os.getcwd(),SURROGATE_PATH,
                                                                f"RF_reg_intrusion_{dd}D.skops") , 
                                                    trusted=True) for dd in dimensions]
        
        models_sea:List[GridSearchCV] = [load(os.path.join(os.getcwd(),SURROGATE_PATH,
                                                           f"RF_reg_sea_{dd}D.skops") , 
                                            trusted=True) for dd in dimensions]

        models_dual:List[GridSearchCV] = [load(file=os.path.join(os.getcwd(),SURROGATE_PATH,
                                                                 f"RF_reg_dual_{dd}D.skops"),
                                                        trusted=True) for dd in dimensions]


    triggers:list = [
            ioh.logger.trigger.ALWAYS
        ]


    logger_ioh_cont_1:List[ioh.logger.Analyzer] = []
    logger_ioh_cont_2:List[ioh.logger.Analyzer] = []

    funct_container_1:List[Starbox_problem] = []
    funct_container_2:List[Starbox_problem] = []

    funct_ioh_cont_RS:List[ioh.logger.Analyzer] = []
    funct_container_RS:List[Starbox_problem] = []


    for idx,dim in enumerate(dimensions):

        if cma_case_handler ==1:
            extra_str = "Hansen"
        elif cma_case_handler ==2:
            extra_str = "Vermetten-De_Nobel"


        logger_ioh = ioh.logger.Analyzer(
            root=os.getcwd(),                  # Store data in the current working directory
            folder_name=os.path.join(os.getcwd(),RESULTS_PATH,str(pid_id),f"Star_Box_CMA_ES_{dim}D_dual"),       # in a folder named: 'my-experiment'
            algorithm_name= "CMA-ES",    # meta-data for the algorithm used to generate these results
            algorithm_info= extra_str,
            store_positions=True,               # store x-variables in the logged files
            triggers= triggers,
        )


        logger_ioh_2 = ioh.logger.Analyzer(
            root=os.getcwd(),                  # Store data in the current working directory
            folder_name=os.path.join(os.getcwd(),RESULTS_PATH,str(pid_id),f"Star_Box_CMA_ES_{dim}D_iso_dual"),       # in a folder named: 'my-experiment'
            algorithm_name=f"CMA-ES_iso",    # meta-data for the algorithm used to generate these results
            algorithm_info= extra_str,
            store_positions=True,               # store x-variables in the logged files
            triggers= triggers,
        )

  
        logger_ioh_RS = ioh.logger.Analyzer(
                folder_name=os.path.join(os.getcwd(),RESULTS_PATH,str(pid_id),f"Star_Box_CMA_ES_{dim}D_RS"),       # in a folder named: 'my-experiment'
                algorithm_name="Random-Search",    # meta-data for the algorithm used to generate these results
                algorithm_info= "Copy from IOH Repository",
                store_positions=True,               # store x-variables in the logged files
                triggers= triggers,
            )

    
        if dim <=5:
            n_vars = 5
        else:
            n_vars = dim

        if dim_case == 1:
            # ff:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,n_variables=5,instance=1,
            #                                                      objective_type=DEFAULT_OBJECTIVE_TYPE)
            # ff2:ioh.problem.RealSingleObjective = Starbox_problem(model_intrusion,model_sea,n_variables=5,instance=2,
            #                                                       objective_type=DEFAULT_OBJECTIVE_TYPE)
            
            ff:Starbox_problem = Starbox_problem(model_dual,None,n_variables=n_vars,instance=2*idx+1,
                                                                objective_type=obj_type,
                                                                lamda=lamda_val,opt_dimensionality=dim,
                                                                intrusion_ref=intrusion)
            ff2:Starbox_problem = Starbox_problem(model_dual,None,n_variables=n_vars,instance=2*idx+2,
                                                                objective_type=obj_type,opt_dimensionality=dim,
                                                                lamda=lamda_val,
                                                                intrusion_ref=intrusion)
            if append_random_search:
                ff_rs:Starbox_problem = Starbox_problem(model_dual,None,n_variables=n_vars,instance=2*idx+4096,
                                                                    objective_type=obj_type,
                                                                    opt_dimensionality=dim,
                                                                    lamda=lamda_val,
                                                                    intrusion_ref=intrusion)
        elif dim_case ==2:

            ff:Starbox_problem = Starbox_problem(models_intrusion[idx],models_sea[idx],n_vars,2*idx+1,
                                                 opt_dimensionality=dim,lamda=lamda_val,
                                                 intrusion_ref=intrusion)
            ff2:Starbox_problem = Starbox_problem(models_intrusion[idx],models_sea[idx],n_vars,2*idx+2,
                                                  opt_dimensionality=dim,lamda=lamda_val,
                                                  intrusion_ref=intrusion)

            if append_random_search:
                ff_rs:Starbox_problem = Starbox_problem(models_intrusion[idx],models_sea[idx],
                                                        n_variables=n_vars,instance=2*idx+4096,
                                                        objective_type=obj_type,
                                                        lamda=lamda_val,
                                                        intrusion_ref=intrusion)
    

        # Append the problems to the list
        funct_container_1.append(ff)
        funct_container_2.append(ff2)

        logger_ioh_cont_1.append(logger_ioh)
        logger_ioh_cont_2.append(logger_ioh_2)

        if append_random_search:
            if dim in random_search_dimensions:
                funct_ioh_cont_RS.append(logger_ioh_RS)
                funct_container_RS.append(ff_rs)
            else:
                funct_ioh_cont_RS.append(None)
                funct_container_RS.append(None)

    # Have some new space...
    del logger_ioh, logger_ioh_2, logger_ioh_RS

    # Logger
    #logger:cma.CMADataLogger= cma.CMADataLogger()
    #logger_2:cma.CMADataLogger= cma.CMADataLogger()

    bestever = cma.optimization_tools.BestSolution()
    bestever_2 = cma.optimization_tools.BestSolution()

    for run in range(n_runs):

        # Set the initial seed
        np.random.seed( int(random_seed+run) )
        c_maes.utils.set_seed(int(random_seed+run) )
        
        if np.all(np.isnan(x0_)):
            x0 = np.random.uniform(-5,5,size=(1,max(dimensions)))
        else:
            x0 = x0_.copy()
        
        print("Testing this array", x0[0,0:5], flush=True)

        
        
        for idx, dim in enumerate(dimensions):


            mu_0, lamda_0 = return_initial_mu_lamda(dim=dim,case_=pop_case)
            
            # Update the dimensionality track for IOH
            funct_container_1[idx].opt_dimensionality = dim
            funct_container_2[idx].opt_dimensionality = dim

            #ff.lamda = lamda_val
            #ff2.lamda = lamda_val


            

            if run == 0:

                # Set the tracks for the logger (Namely intrusion and sea)
                logger_ioh_cont_1[idx].watch(funct_container_1[idx],['cur_intrusion','cur_sea'])
                logger_ioh_cont_1[idx].add_run_attributes(funct_container_1[idx],['opt_dimensionality','lamda'])

                
                logger_ioh_cont_2[idx].watch(funct_container_2[idx],['cur_intrusion','cur_sea'])
                logger_ioh_cont_2[idx].add_run_attributes(funct_container_2[idx],['opt_dimensionality','lamda'])

                funct_container_1[idx].attach_logger(logger_ioh_cont_1[idx])
                funct_container_2[idx].attach_logger(logger_ioh_cont_2[idx])


            if cma_case_handler == 1:


                opts = return_simulation_setup(x0,random_seed+run,dim,initial_budget=budget,full_sampler=True)
                opts2 = return_simulation_setup(x0,random_seed+run,dim,initial_budget=budget,full_sampler=False)


                try:
                    res_1:cma.evolution_strategy.CMAEvolutionStrategyResult = cma.fmin(funct_container_1[idx],
                                                                                       x0=x0.flatten().tolist(),
                                                                                       sigma0=sigma_0,
                                                                                       options=opts,
                                                                                       restarts=max_restarts,
                                                                                       eval_initial_x=True)
                    
                    res_2:cma.evolution_strategy.CMAEvolutionStrategyResult = cma.fmin(funct_container_2[idx],
                                                                                       x0=x0.flatten().tolist(), 
                                                                                       sigma0=sigma_0,
                                                                                       options=opts2,
                                                                                       eval_initial_x=True,
                                                                                       restarts=max_restarts)
                
                except ValueError as e:
                    raise ValueError(e.args,"There was the same error")
                
                

            elif cma_case_handler==2:


                # Evaluate the functions
                #ff(x0.ravel())
                #ff2(x0.ravel())

                # Get the adjusted initial input
                x01 = adjust_initial_input(x0=x0,dim=dim)

                if dim_case == 1:
                    @modify_array(x0)
                    def obj_func_1(x_, func:Starbox_problem=funct_container_1[idx]) -> float:
                        return func(x_)
                    
                    @modify_array(x0)
                    def obj_func_2(x_, func:Starbox_problem=funct_container_2[idx]) -> float:
                        return func(x_)
                    
                    # Evaluate the first point
                    obj_func_1(x01)
                    obj_func_2(x01)
                
                elif dim_case ==2:

                    def obj_func_1(x_, func:Starbox_problem=funct_container_1[idx]) -> float:
                        return func(x_)
                    
                    def obj_func_2(x_, func:Starbox_problem=funct_container_2[idx]) -> float:
                        return func(x_)
                    
                    # Evaluate the first point
                    obj_func_1(x01)
                    obj_func_2(x01)


                module_1 = return_simulation_setup_2(max_restarts,
                                                     full_sampler=True)
                
                module_2 = return_simulation_setup_2(max_restarts,
                                                     full_sampler=False)

                #params_mod_1:Parameters = Parameters()
                #cma_es__:ModularCMAES = ModularCMAES(fitness_func=obj_func_1,
                #                                     budget=budget,dim=dim)
                
                #cma_es__.parameters.fopt

                
                opts = c_maes.parameters.Settings(dim=dim, modules=module_1,x0=x01, 
                                                sigma0 =sigma_0, budget=budget, verbose=verbose,
                                                mu0 = mu_0, lambda0 = lamda_0,
                                                lb=np.array([-5.0]*dim), 
                                                ub=np.array([5.0]*dim))
                
                opts2 = c_maes.parameters.Settings(dim=dim, modules=module_2,x0=x01, 
                                                sigma0 =sigma_0, budget=budget, verbose=verbose,
                                                mu0 = mu_0, lambda0 = lamda_0,
                                                lb=np.array([-5.0]*dim), 
                                                ub=np.array([5.0]*dim))

                parameters:c_maes.Parameters = c_maes.Parameters(opts)
                parameters2 = c_maes.Parameters(opts2)

                

                cma1 = c_maes.ModularCMAES(parameters)
                cma2 = c_maes.ModularCMAES(parameters2)

                # Attach the tracking of mu and sigma to the logger
                logger_ioh_cont_1[idx].watch(parameters,['mu[0]'])
                logger_ioh_cont_2[idx].watch(parameters2,['mu[0]'])

                cma1.run(obj_func_1)
                #cma1.run(ff)
                #while not cma1.break_conditions():
                #    cma1.step(ff)
                cma2.run(obj_func_2)

            else:
                raise NotImplementedError("No other libraries besides hansen's and de nobel's are used")

            # Run Random Search
            if append_random_search:
                if funct_container_RS[idx] is not None:
                    
                    funct_ioh_cont_RS[idx].watch(funct_container_RS[idx],['cur_intrusion','cur_sea'])
                    funct_ioh_cont_RS[idx].add_run_attributes(funct_container_RS[idx],['opt_dimensionality','lamda'])
                    
                    # Attach
                    if run ==0:
                        funct_container_RS[idx].attach_logger(funct_ioh_cont_RS[idx])
                    
                    @modify_array(x0)
                    def obj_func_RS(x_, func:Starbox_problem=funct_container_RS[idx]) -> float:
                        return func(x_)
                    # Evaluate the function
                    obj_func_RS(x01)

                    # Run the Random Search
                    rs_obj:RandomSearch = RandomSearch(budget=budget,dim=dim)

                    rs_obj(func=obj_func_RS)

                    funct_container_RS[idx].reset()

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

            funct_container_1[idx].reset()
            funct_container_2[idx].reset()


        

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        prog="StarBox_Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Instantiate an optimization run of the Star Box Crash 
            --------------------------------
            Generate samples for evaluation by a real world function / simulator
                > python(3) test_opt_general_multi_dummy_2.py --pid 42
            """
        ),
    )
    
    parser.add_argument(
        "--budget",
        "-b",
        default=BUDGET,
        type=int,
        help="Budget of the optimization loop",
    )


    parser.add_argument(
        "--pop_case",
        "-c",
        default=POP_CASE,
        choices= [0,1,2,3],
        type=int,
        help="Population Case",
    )

    parser.add_argument(
        "--n_runs",
        "-nr",
        default=N_RUNS,
        type=int,
        help="Number of Runs",
    )

    parser.add_argument(
        "--dimensionality_case",
        "-dc",
        default=DIMENSIONALITY_CASE,
        type=int,
        choices= [1, 2],
        help="A handle to switch between the full space surrogate or use subspace surrogate",
    )

    parser.add_argument(
        "--cma_es_lib",
        "-ll",
        default=CMA_ES_HAND,
        choices= [1,2],
        type=int,
        help="A handle to switch between the implementation of Niko Hansen or LIACS implementation of CMA-ES",
    )

    parser.add_argument(
        "--max_restarts",
        "-mr",
        default=MAX_RESTARTS_DEFAULT,
        type=int,
        help="Set the number of maximum restarts",
    )

    parser.add_argument(
        "--objective_type",
        "-ot",
        default=DEFAULT_OBJECTIVE_TYPE,
        type=int,
        help="A handle to switch the type of objective",
        choices= [1,2],
    )

    parser.add_argument(
        "--lamda",
        "-l",
        default=LAMDA,
        type=float,
        help="The penalty factor used for the objective function",
    )

    parser.add_argument(
        "--delta_prime",
        "-dp",
        default=INTRUSION_PRIME,
        type=float,
        help="The reference intrusion used for the objective function",
    )

    parser.add_argument(
        "--sigma_0",
        "-s0",
        default=SIGMA_0,
        type=float,
        help="The initial step size for CMA-ES",
    )

    parser.add_argument(
        "--pid",
        "-p",
        type=int,
        help="A handle to switch the type of objective",
        default=np.random.randint(low=1, high=1024)
        #required=True
    )

    parser.add_argument(
        "--verbose",
        "-v",
        default=False ,
        type=int,
        help="A handle to disable verbose output from optimizer",
    )

    parser.add_argument(
        "--random_seed",
        "-rs",
        default=SEED,
        type=int,
        help="A seed to kickstart the randomness",
    )

    parser.add_argument(
        "--dimensions",
        "-d",
        default= DIMENSIONS,
        type= int,
        nargs = "+",
        help= "Set the number of dimensions to be analyzed"
    )

    parser.add_argument(
        "--initial_point",
        "-x0",
        default= np.full(shape=(1,5),fill_value=np.nan),
        type= float,
        nargs = 5,
        help= "Set the an initial point to check some convergence"
    )

    parser.add_argument(
        "--append_random_search",
        "-ars",
        default= False,
        type= bool,
        help= "Append random search"
    )

    parser.add_argument(
        "--dimensions_random_search",
        "-drs",
        default= None,
        type= int,
        nargs = "*",
        help= "Append dimensions of random search"
    )

    args = parser.parse_args()

    # Set unique values (this is due that the seed is the same so it will be very similar)
    args.dimensions = sorted(list(set(args.dimensions)))
    args.append_random_search = bool(args.append_random_search)

    print(args.append_random_search,flush=True)

    # Perform some checks on the list
    for elem in args.dimensions:
        if elem not in DIMENSIONS and elem <=5:
            raise ValueError(f"The element {elem} is not part of the set (1,3,5)")
        
        if elem == 1 and args.cma_es_lib == 1:
            warnings.warn("Switching to De Nobel-Vermetten implementation of CMA-ES")
            args.cma_es_lib == 2

    if args.append_random_search:
        if args.dimensions_random_search is not None:
            args.dimensions_random_search = sorted(list(set(args.dimensions_random_search)))
            for idx,elem in enumerate(args.dimensions_random_search):
                if elem not in args.dimensions:
                    args.dimensions_random_search.pop(idx)
        else:
            raise ValueError("Set some dimensions to run random search!")
                    
    # Do a loop to check the dimensions
    # Just set unique dimensions values

    print("The initial step size is:", args.sigma_0)
    print("The running dimensions are", args.dimensions,flush=True)
    print("The running seed is:", args.random_seed, flush= True)

    if args.append_random_search:
        print("The dimensions to run random search are:", args.dimensions_random_search)


    main(pid_id=args.pid,
         dim_case=args.dimensionality_case,
         cma_case_handler=args.cma_es_lib,
         budget= args.budget,
         verbose=args.verbose,
         lamda_val=args.lamda,
         pop_case=args.pop_case,
         obj_type=args.objective_type,
         max_restarts=args.max_restarts,
         n_runs=args.n_runs,
         sigma_0=args.sigma_0,
         intrusion = args.delta_prime,
         random_seed = args.random_seed,
         dimensions= args.dimensions,
         x0_= np.array(args.initial_point).reshape((1,5)),
         append_random_search=args.append_random_search,
         random_search_dimensions=args.dimensions_random_search)