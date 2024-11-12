import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json


class IOH_Parser:
    

    def __init__(self, filepath:str):
        

        # Load the file to parse the information
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        
        
        except FileNotFoundError as e:
            # Raise this error in case the file is not found
            raise(e.args)
        
        except json.JSONDecodeError as e:

            raise(e.args)
        
        #except Exception as e:
        #    raise(e.args)
        
        # Get the current directory
        curdir = os.path.dirname(filepath)

        self.__json_file:str = filepath
        # Fill the parser
        self.__algorithm:Algor = Algor(data['algorithm'])
        self.__version = data['version']
        self.__function_id = data['function_id']
        self.__function_name = data['function_name']
        self.__maximization = data['maximization']
        self.__attributes = data["attributes"]
        self.__scenarios:list = [Opt_Scenario(a,curdir) for idx,a in enumerate(data["scenarios"])]


    def reload(self):
        # Load the file to parse the information
        try:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
        
        
        except FileNotFoundError as e:
            # Raise this error in case the file is not found
            raise(e.args)
        
         # Get the current directory
        curdir = os.path.dirname(self.json_file)

        # Fill the parser
        self.__algorithm:Algor = Algor(data['algorithm'])
        self.__version = data['version']
        self.__function_id = data['function_id']
        self.__function_name = data['function_name']
        self.__maximization = data['maximization']
        self.__attributes = data["attributes"]
        self.__scenarios:list = [Opt_Scenario(a,curdir) for idx,a in enumerate(data["scenarios"])]


    @property
    def json_file(self)-> str:
        return self.__json_file
    
    @property
    def version(self)-> str:
        return self.__version
    
    @property
    def function_id(self)-> int:
        return self.__function_id
    
    @property
    def function_name(self)-> int:
        return self.__function_name
    
    @property
    def maximization(self)-> bool:
        return self.__maximization
    
    @property
    def attributes(self)-> dict:
        return self.__attributes
    
    @property
    def algorithm_name(self)->str:
        return self.__algorithm.name
    
    @property
    def algorithm_info(self)->str:
        return self.__algorithm.info
    
    @property
    def number_of_instances(self)->int:
        return len(self.__scenarios)
    

    def return_complete_table_per_instance(self,instance_id:int):
        if instance_id < 0 or instance_id > self.number_of_instances -1:
            raise("The position is out of bounds!")
        
        return self.__scenarios[instance_id].data
    
    def get_the_best_score_per_instance(self,instance_id:int):
        if instance_id < 0 or instance_id > self.number_of_instances -1:
            raise(ValueError())
        
        if self.maximization:
            return self.__scenarios[instance_id].data.groupby("run").max()
        else:
            return self.__scenarios[instance_id].data.groupby("run").min()

    

class Algor:
    def __init__(self,algorithm_dict:dict):
        self.__name = algorithm_dict['name']
        self.__info = algorithm_dict['info']

    
    @property
    def name(self)->str:
        return self.__name
    
    @property
    def info(self)->str:
        return self.__info


class Opt_Scenario:
    def __init__(self,scenarios_dict:dict, directory:str) -> None:
        self.__dimensions:int = scenarios_dict['dimension']
        self.__path:str = os.path.join(directory, scenarios_dict['path'])
        self.__runs:list = [Runs(scenarios_dict['runs'][idx]) for idx,a in enumerate(scenarios_dict["runs"])]
        self.__num_runs:int = len(scenarios_dict['runs'])
        self.__data:pd.DataFrame = self.__load_dat_file_with_iterations()
        
        
    
    @property
    def dimensions(self)->int:
        return self.__dimensions

    @property
    def path(self)->str:
        return self.__path
    
    @property
    def data(self)->pd.DataFrame:
        return self.__data
        
    def best_per_run(self)-> pd.DataFrame:
        return self.__best_per_run
    

    def __load_dat_file_with_iterations(self)->pd.DataFrame:

        "Use the default pandas import"
        try:
            df:pd.DataFrame = pd.read_csv(self.path, delimiter=" ")
        except FileNotFoundError as e:
            raise("The file was not found", e.args)
        except Exception as e:
            raise("Cannot trace the source of error", e.args)

        if df.dtypes.iloc[0] == pd.StringDtype:
            boolean_elims = df.evaluations.str.contains('evaluations') == False
        else:
            boolean_elims = np.ones(df.evaluations.size,dtype=bool)


        init_run:int = 1
        list_:np.ndarray = np.zeros(boolean_elims.size, dtype=int)
        for a,b in enumerate(boolean_elims):
            if b == True:
                list_[a] = init_run
            else:
                init_run += 1
                list_[a] = init_run
        
        df = df.assign(run=list_)

        cols = list(df.columns)
        cols = [cols[-1] ] + cols[0:-1] 

        # Eliminate the repeated headers
        df = df[boolean_elims]
        df = df[cols]

        # Convert all the instances to floats
        for idx,col in enumerate(df.columns):
            if col == 'raw_y' or (col.startswith('x') and len(col.split('x'))==2): 
                df[col] = df[col].astype(float)
            elif col== 'evaluations':
                df[col] = df[col].astype(int)
            elif col== 'cur_sea' or col=="cur_intrusion":
                df[col] = df[col].astype(float)

        df.rename(columns={"raw_y":"Objective"},inplace=True)

        return df
    
class Runs:
    def __init__(self,runs_dict:dict) -> None:
        # Normal parameters
        self.__instance:int = runs_dict['instance']
        self.__tot_evals:int = runs_dict['evals']
        
        #Obtain the dictionary with the results
        self.__best:dict = runs_dict['best']
        
    
    
    @property
    def instance(self)->int:
        return self.__instance
    
    @property
    def evals(self)->int:
        return self.__tot_evals