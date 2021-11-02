#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from functools import reduce
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
# array([  25,   28,   35,  102, 1289, 2082, 2684, 2698, 2706, 2922, 2966, 3002, 3003, 3005])

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
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
# print(df.tail())
# # temperatura: Outliers > 2021-08-15
# # temperatura: Missing: index<60000
# # precipitacion: Missing:  index>180.000
df=df.iloc[0:190000]
df=df.loc[df['fecha'] > "2019-03-01",:]

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
print(df.tail())
df.to_csv(df_path, header=True, index=False, na_rep="NaN")

# %%
# Ubicaciion 1289
df_path = Path("/home/diego/weather-control/data/processed/1289.csv")
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
# Ubicaciion 2082
df_path = Path("/home/diego/weather-control/data/processed/2082.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
#df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
#df.set_index('fecha', drop=False, inplace=True)


print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df = df.iloc[2000:]
#outliers de temperatura -15.5 los elimino
df.loc[df['temperatura'] == df['temperatura'].min(),['hr','precipitacion','temperatura']] = np.nan
df.interpolate(inplace=True)
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
print(df.isna().sum())
plt.show()
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 2684
df_path = Path("/home/diego/weather-control/data/processed/2684.csv")
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
# Ubicaciion 2698
df_path = Path("/home/diego/weather-control/data/processed/2698.csv")
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
# Ubicaciion 2706
df_path = Path("/home/diego/weather-control/data/processed/2706.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

df=df.loc[(df['fecha'] > "2020-02-27") & (df['fecha'] <= "2021-08-11") , :]

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df.iloc[:].loc[:,'temperatura'])
ax[1].plot(df.iloc[:].loc[:,'hr'])
ax[2].plot(df.iloc[:].loc[:,'precipitacion'])

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
# %%
# Ubicaciion 2966
df_path = Path("/home/diego/weather-control/data/processed/2966.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

# hr cosas raras en <9000
# missing al final y antes del 11000
print(df.iloc[11000:46000].isna().sum())
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df.iloc[11000:46000].loc[:,'temperatura'])
ax[1].plot(df.iloc[11000:46000].loc[:,'hr'])
ax[2].plot(df.iloc[11000:46000].loc[:,'precipitacion'])
df = df.iloc[11000:46000]
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 3002
df_path = Path("/home/diego/weather-control/data/processed/3002.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
# # valores raros de temperatura
# df.loc[df['temperatura'] == df['temperatura'].min(), ['hr','precipitacion','temperatura']] = np.nan
# df.interpolate(inplace=True)
df = df.loc[df['fecha']> "2021-05-01"]

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 3003
df_path = Path("/home/diego/weather-control/data/processed/3003.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())

fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df = df.loc[df['fecha']> "2021-05-06"]
# df.loc[df['temperatura'] == df['temperatura'].min(),:] = np.nan
# df.interpolate(inplace=True)
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
# Ubicaciion 3005
df_path = Path("/home/diego/weather-control/data/processed/3005.csv")
df = pd.read_csv(df_path, na_values="NaN", header=0)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
df.set_index('fecha', drop=False, inplace=True)

print(df.isna().sum())
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()

df = df.loc[df['fecha']>"2021-05-01", :]
fig, ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(df['temperatura'])
ax[1].plot(df['hr'])
ax[2].plot(df['precipitacion'])
plt.show()
df.to_csv(df_path, header=True, index=False, na_rep="NaN")
# %%
