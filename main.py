import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

#Primero tenemos que cargar nuestro dataset
USA_Housing = pd.read_csv("USA_Housing.csv")


print("Introducción")
print(USA_Housing.columns)
#Se utiliza para cargar las columnas de nuestro dataset
#Algunas de las columnas que contiene son: Price, Address, Area Population ...

print("VARIABLES + DATOS")
print(USA_Housing.head())
#Se utiliza para representar visualmente las columnas anteriores.
#Cada fila será un dato y cada columa será cada una de las características que emplearemos  y observaremos si existen relaciones entre ellas.
print("Descripción del DF")
print(USA_Housing.describe())
#En esta tabla se muestra cuantos valores hay de cada uno, la media en cada variable, la desviación, mínimos, máximos, los cuartiles.
#Observamos que la media de las personas que viven en un area es de 36163, el número mínimo de personas que vivirán en una zona es de 172 personas y el número máximo es de 69621. La mediana de este grupo es de 36199 personas, el primer cuartil es de 29403 y el tercer cuartil es de 42861.

print("Información del DF")
print(USA_Housing.info())

#VARIABLES CATEGÓRICAS: Address
#VARIABLES NUMÉRICAS: Price, Area Population, Avg. Area Number of Bedrooms,Avg. Area Number of Rooms, Avg. Area House Age,Avg. Area Income.
#Addressnsupuestamente es categórica, pero phyton no lo detecta como tal, ya que es como si fuese el ID de la calle.

#A) ANÁLISIS DE VARIABLES CATEGÓRICAS
  #ANÁLISIS UNIVARIADO
def bar_plot (variable):
  var= USA_Housing[variable]
  varValue = var.value_counts()

  #visualización
  plt.figure(figsize =(9,3))
  plt.bar (varValue.index, varValue)
  plt.xticks (varValue.index, varValue.index.values)
  plt.ylabel("Frecuencia")
  plt.title (variable)
  plt.show()

  print("{}:\n{}".format(variable, varValue))

#Como no hay variables categóricas esta función no se va a realizar

#VARIABLES NUMÉRICAS
category2 = ["Price", "Area Population", "Address"]
for c in category2:
    print("{} \n".format(USA_Housing[c].value_counts()))

#Analisis de variables numéricas 
  #Analisis univariado
def plot_hist(variable):
  plt.figure(figsize = (9,3))
  plt.hist(USA_Housing[variable], bins = 50)
  plt.xlabel(variable)
  plt.ylabel("Frecuencia")
  plt.title("Distribución variable {} con histograma".format(variable))
  plt.show()

numericVar = ["Avg. Area Income","Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms","Price", "Area Population"]
for n in numericVar:
  plot_hist(n)
#El histograma de Avg. Area Income observamos que muestra una campana de Gauss, es decir, nos encontramos ante un gráfico aparentemente simétrico.
#Se puede observar que hay una gran frecuencia en las casas construidas hace 5 y 7 años, aunque, en el año 6 anotamos un notable descenso de casas construidas.
#Por otro lado, la media del número de habitaciones compartidas que tienen las casas está etre 7 y 8, aunque mayormente se tienen 8 habitaciones.
#Respecto al número de habitaciones personales las casas tienen mayormente 3 habitaciones, aunque también hay una gran tendencia en tener 4 habitaciones individuales.
# La media de personas que viven en un area es de 40000 y la máximas personas que viven en un area es de 700000.
  
#Al no haber variables categóricas no va a salir nada
def plot_3chart(df, feature):
  import matplotlib.gridspec as gridspec
  from matplotlib.ticker import MaxNLocator
  from scipy.stats import norm
  from scipy import stats

  #Crear un gráfico personalizado
  fig = plt.figure(constrained_layout=True, figsize=(12, 8))
  #Crear una cuadrícula de tres columnas y tres filas
  grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
  #Crear un histograma
  ax1 =fig.add_subplot(grid[0, :2])
  #Establecer el título
  ax1.set_title('Histogram')
  #Trazar el histograma
  sns.distplot(df.loc[:, feature],
               hist=True,
               kde=True,
               fit=norm,
               ax=ax1,
               color='#e74c3c')
  ax1.legend(labels=['Normal', 'Actual'])

  ax2 = fig.add_subplot(grid[1, :2])
  ax2.set_title('Probability Plot')
  #Customizar el gráfico QQ
  stats.probplot(df.loc[:, feature].fillna(np.mean(USA_Housing.loc[:, feature])),
                plot=ax2)
  ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
  ax2.get_lines()[0].set_markersize(12.0)
  
  #Customizar el digrama de cajas y bigotes
  ax3 = fig.add_subplot(grid[:, 2])
  #Establecer el título
  ax3.set_title('Box Plot')
  #Trazar el diagrama de cajas y bigotes
  sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')
  ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))
  
  plt.suptitle(f'{feature}', fontsize=24)

plot_3chart(USA_Housing, 'Area Population')

plot_3chart(USA_Housing, 'Price')

plot_3chart(USA_Housing, 'Avg. Area Number of Bedrooms')

plot_3chart(USA_Housing, "Avg. Area Number of Rooms")

plot_3chart(USA_Housing, "Avg. Area House Age")

plot_3chart(USA_Housing,"Avg. Area Income")


  
USA_Housing[["Area Population", "Price"]].groupby(["Area Population"], as_index = False).mean().sort_values(by="Price",ascending = False)

#Detección de outliers(Valores atípicos)
def detect_outliers(df,features):
  outlier_indices = []
  for c in features:
    #1º Cuartil
    Q1 = np.percentile(df[c],25)
    #3º Cuartil
    Q3 = np.percentile(df[c],75)
    #Rango intercuartílico
    IQR = Q3 - Q1
    #Outlier step
    outlier_step = IQR * 1.5
    #Detectar valores atípicos y sus índices
    outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
    #Almacenar indices
    outlier_indices.extend(outlier_list_col)

  outlier_indices = Counter(outlier_indices)
  multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
  return multiple_outliers

USA_Housing.loc[detect_outliers(USA_Housing,["Area Population", "Price"])]
USA_Housing = USA_Housing.drop(detect_outliers(USA_Housing,["Area Population", "Price"]),axis = 0).reset_index(drop = True)



#Valores perdidos
USA_Housing_len = len(USA_Housing)
USA_Housing.head()

#Encontrar valores perdidos
sns.heatmap(USA_Housing.isnull(),
           yticklabels=False,
           cbar=False,
           cmap='magma')

plt.title('Valores perdidos en conjunto de train')

plt.xticks(rotation=90)
plt.show()

#Como se puede observar en los gráficos no hay ningún valor perdido ya que no hay ninguna ralla en amarillo.

corr = USA_Housing.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5, fmt = '.2f',ax=ax)
plt.show()

#Respecto a la correlación observamos que el precio y la Avg. Area Income son las variables que más relación tienen entre sí ya uqe tienen una correlación muy cercana a 1. El resto de variables no están relacionadas entre sí ya que tienen una correlación muy cercana a 0.

#Price--Area Population
g = sns.factorplot(x = "Price", y = "Area Population", data = USA_Housing, kind = "bar", size = 6)
g.set_ylabels("Relación entre el precio y lo concurrida que está la zona")
plt.show()

#Price--Avg. Area Number of Rooms
g = sns.factorplot(x = "Price", y = "Avg. Area Number of Rooms", data = USA_Housing, kind = "bar", size = 6)
g.set_ylabels("Relación entre el precio y el número de habitaciones")
plt.show()





  
