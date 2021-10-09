# %%
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

with open('/home/diego/weather-control/data/outputs/experimento7_72-72_Sum0505_model2_lr1e-2-1e-3nosche/prediction.pickle', 'rb') as handler:
    predicciones = pickle.load(handler)

MUESTRA = 32
for _ in predicciones.iloc[MUESTRA].loc['Y']:
    plt.plot(_, 'b')

plt.plot(np.mean(predicciones.iloc[MUESTRA].loc['Y'], axis=0).reshape(-1,1), 'g')
plt.plot(np.mean(predicciones.iloc[MUESTRA].loc['Ypred'], axis=0).reshape(-1,1), 'r')
# %%
