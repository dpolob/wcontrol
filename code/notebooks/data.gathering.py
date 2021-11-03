#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path
from functools import reduce
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


def precipitacionmenorzero(df: pd.DataFrame=None) -> pd.DataFrame:
    df.loc[df.precipitacion <0, ['precipitacion']] = 0
    return df



raw_path = Path('/home/diego/weather-control/data/raw/sql-raw/raw.csv')
raw = pd.read_csv(raw_path, header=0, na_values='\0',sep=";")

ubicaciones = raw['ubicacion'].unique()
print(f"Ubicaciones disponibles {raw['ubicacion'].unique()}")

metricas = raw['metrica'].unique()
for ubicacion in tqdm(ubicaciones, desc='Ubicaciones', leave=True):
    dfs = []
    for metrica in tqdm(metricas, desc='Metricas', leave=True):
        a = raw.loc[(raw.ubicacion == ubicacion) & (raw.metrica == metrica), ['fecha', 'valor']]
        if metrica == 1:
            a.rename(columns = {'valor':'temperatura'}, inplace = True)
        if metrica == 6:
            a.rename(columns = {'valor':'hr'}, inplace = True)
        if metrica == 11:
            a.rename(columns = {'valor':'precipitacion'}, inplace = True)
        dfs.append(a)
        dfs.sort(key=lambda x: len(x), reverse=True)
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on='fecha',  how='outer'), dfs)
    output_path = Path(f"/home/diego/weather-control/data/processed/{ubicacion}.csv")
    df_merged.to_csv(output_path, header=True, index=False, na_rep="NaN")
# %%
# array([  25,   28,   35,   1289, 2082, 2684, 2698, 2706, 2922, 2966])
#####################
### ELIMINO 102, 3002, 3003 y 3005
########################
# Ubicaciion 25
df_path = Path("/home/diego/weather-control/data/processed/25.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)
print(df.isna().sum())

fechas = df['fecha']
df.drop(columns=['fecha'], inplace=True)
imp_mean = IterativeImputer(random_state=0)
idf = imp_mean.fit_transform(df)
idf = pd.DataFrame(idf, columns=df.columns)
df['temperatura'] = idf['temperatura'].values
df['hr'] = idf['hr'].values
df['precipitacion'] = idf['precipitacion'].values
df['fecha'] = fechas
print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
# %%
# Ubicaciion 28
df_path = Path("/home/diego/weather-control/data/processed/28.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
# %%
# Ubicaciion 35
df_path = Path("/home/diego/weather-control/data/processed/35.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())
print(df[df.isna().any(axis=1)])
nanas=df[df.isna().any(axis=1)].index

fechas = df['fecha']
df.drop(columns=['fecha'], inplace=True)
imp_mean = IterativeImputer(random_state=0)
idf = imp_mean.fit_transform(df)
idf = pd.DataFrame(idf, columns=df.columns)
df['temperatura'] = idf['temperatura'].values
df['hr'] = idf['hr'].values
df['precipitacion'] = idf['precipitacion'].values
df['fecha'] = fechas
print(df.isna().sum())

print(df.loc[nanas, :])
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 102
df_path = Path("/home/diego/weather-control/data/processed/102.csv")
os.remove(df_path)
# %%
# Ubicaciion 1289
df_path = Path("/home/diego/weather-control/data/processed/1289.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
# borrar las tres ultimas
df = df.iloc[:-3,:]
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fechas = df['fecha']
df.drop(columns=['fecha'], inplace=True)

imp_mean = IterativeImputer(random_state=0)
idf = imp_mean.fit_transform(df)
idf = pd.DataFrame(idf, columns=df.columns)
df['temperatura'] = idf['temperatura'].values
df['hr'] = idf['hr'].values
df['precipitacion'] = idf['precipitacion'].values
df['fecha'] = fechas
print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

#df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2082
df_path = Path("/home/diego/weather-control/data/processed/2082.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df = df.iloc[3000:]
#outliers de temperatura -15.5 los elimino
df.loc[df['temperatura'] == df['temperatura'].min(),['hr','precipitacion','temperatura']] = np.nan

df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
fechas = df['fecha']
df.set_index('fecha', drop=True, inplace=True)


print(df.isna().sum())
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
idf = imp_mean.fit_transform(df.values)
idf = pd.DataFrame(idf, columns=df.columns)
df['temperatura'] = idf['temperatura'].values
df['precipitacion'] = idf['precipitacion'].values
df['hr'] = idf['hr'].values

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df['fecha'] = df.index
df = precipitacionmenorzero(df)
df.to_csv(df_path, header=True, index=False, na_rep="NaN")

# %%
# Ubicaciion 2684
df_path = Path("/home/diego/weather-control/data/processed/2684.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df['fecha'] = df['fecha'].dt.strftime('%Y-%m-%d %H:%M')
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M')
df.set_index('fecha', drop=True, inplace=True)
print(df)
print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df['fecha'] = df.index
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2698
df_path = Path("/home/diego/weather-control/data/processed/2698.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df['fecha'] = df['fecha'].dt.strftime('%Y-%m-%d %H:%M')
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2706
df_path = Path("/home/diego/weather-control/data/processed/2706.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
# borrar las 25 ultimas
df = df.iloc[:-15,:]
print(df.tail(25))


df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

df=df.loc[(df['fecha'] > "2020-02-27") & (df['fecha'] <= "2021-08-11") , :]
fechas = df['fecha']
df.drop(columns=['fecha'], inplace=True)
imp_mean = IterativeImputer(random_state=0)
idf = imp_mean.fit_transform(df)
idf = pd.DataFrame(idf, columns=df.columns)
df['temperatura'] = idf['temperatura'].values
df['hr'] = idf['hr'].values
df['precipitacion'] = idf['precipitacion'].values
df['fecha'] = fechas

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df.iloc[:].loc[:,'temperatura'])
ax[1].plot(df.iloc[:].loc[:,'hr'])
ax[2].plot(df.iloc[:].loc[:,'precipitacion'])

print(df.isna().sum())
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2922
df_path = Path("/home/diego/weather-control/data/processed/2922.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2966
df_path = Path("/home/diego/weather-control/data/processed/2966.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
# borrar las 8 ultimas
df = df.iloc[:-8,:]
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

# # hr cosas raras en <9000
df=df.loc[(df['fecha'] >= "2020-08-10")  , :]
print(df.isna().sum())
print(df.loc[df.precipitacion.isna()].index)

df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 3002
df_path = Path("/home/diego/weather-control/data/processed/3002.csv")
os.remove(df_path)
# %%
# Ubicaciion 3003
df_path = Path("/home/diego/weather-control/data/processed/3003.csv")
os.remove(df_path)
# %%
# Ubicaciion 3005
df_path = Path("/home/diego/weather-control/data/processed/3005.csv")
os.remove(df_path)