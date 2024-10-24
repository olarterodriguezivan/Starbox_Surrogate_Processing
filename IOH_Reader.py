import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IOH_parser import IOH_Parser
import os

BUDGET:int = 200
# Get an address
filep_iso:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES_iso/IOHprofiler_f1121_StarBox.json"
filep:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES/IOHprofiler_f1121_StarBox.json"
#filep2:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES_iso_3-1/IOHprofiler_f1121_StarBox.json"
#filep3:str = "/home/ivanolar/Documentos/Star_Box_CMA_ES_iso_5-1/IOHprofiler_f1121_StarBox.json"

parser = IOH_Parser(filep_iso)
data:pd.DataFrame = parser.return_complete_table_per_instance(0)
data2 = parser.return_complete_table_per_instance(1)
data3 = parser.return_complete_table_per_instance(2)

fig, ax = plt.subplots(1,1)
ax.plot(data['evaluations'].to_numpy(),-1*data['Objective'].ewm(span=4).mean(),label="1D")
ax.plot(data2['evaluations'].to_numpy(),-1*data2['Objective'].ewm(span=7).mean(),label="3D")
ax.plot(data3['evaluations'].to_numpy(),-1*data3['Objective'].ewm(span=8).mean(), label="5D")
ax.set_xlabel("Evaluation Index")
ax.set_ylabel("Objective")
ax.set_xlim(xmin=0,xmax=BUDGET)
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

plt.show()
