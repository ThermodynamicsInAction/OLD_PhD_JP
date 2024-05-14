#Załadowanie odpowiednich bibliotek
import numpy as np
import keras
from scikeras.wrappers import KerasRegressor
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

'''Ważna uwaga: kod można wykonywać metodą od-do w środowisku PyCharm zaznaczając kawałek kodu i wykonując ctrl-alt-e 
Pozwoli to na wykonanie kodu zrobionego PARADYGMATEM FUNKCYJNYM podobnie jak w notatniku jupytera.
'''

#wczytanie I zapis pliku csv do formatu pandas dataframe
df = pd.read_csv('merged_data1.csv',encoding = 'utf8',sep=';')

#Zdefiniowanie zbioru X składającego się z 5 zmiennych wejścia oraz Y będącego zbiorem wyjścia (gęstość/prędkość)
X = df[['M_C', 'M_A', 'IS_SYM', 'P', 'T']].values
y = df[['Rho']].values

#Przygotowanie danych - podzielenie danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

from sklearn.preprocessing import MinMaxScaler
#Metoda służąca do skalowania danych i normalizacji.
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Załadowanie bibliotek służących do budowy modelu sieci.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dense, Dropout
from keras.regularizers import l2

#Tutaj nastepuje budowa sieci
model = Sequential()
model.add(Dense(5, activation='tanh', kernel_regularizer=l2(0.1)))  
model.add(Dense(5, activation='tanh', kernel_regularizer=l2(0.1)))
model.add(Dense(55, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(55, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
#Ostatnia, wyjściowa warstwa
model.add(Dense(1))

#Kompilacja i trenowanie modelu
liczba = 1800
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x=X_train, y=y_train, epochs=liczba)

#Generowanie predykcji dla zbioru testowego i treningowego:
y_test_m=model.predict(X_test)
y_train_m= model.predict(X_train)
# Metryki dla zbioru testowego
r2 = r2_score(y_test_m, y_test)
print(f"Współczynnik determinacji R^2: {r2}")
mse = mean_squared_error(y_test_m, y_test)
print(f"Błąd średniokwadratowy MSE: {mse}")

#Metryki dla zbioru treningowego
r2_train = r2_score(y_train_m, y_train)
print(f"Współczynnik determinacji R^2: {r2_train}")
mse_train = mean_squared_error(y_train, y_train)
print(f"Błąd średniokwadratowy MSE: {mse_train}")
'''Nowy komentarz do kodu- sytuacja gdy zbiór treningowy daje lepsze wyniki niż testowy jest pożądana - jest to pierwszy test czy model nie jest 
przeuczony
'''

#Walidacja krzyżowa:
def create_model():
    return model

keras_regressor = KerasRegressor(build_fn=create_model, epochs=liczba,batch_size= 64)# Ustaw odpowiednie parametry
cv_scores = cross_val_score(keras_regressor, X_test, y_test, cv=5,scoring='r2') #W istocie cv = 20

mean_r2 = cv_scores.mean()
print("Średni R^2 po walidacji krzyżowej:", mean_r2)

'''Sprawdzenie modelu na danych treningowych i testowych
oraz stworzenie dwóch dataframe’ów Pandas '''


model.evaluate(X_test,y_test)
test_predictions = model.predict(X_test)
train_predictions = model.predict(X_train)
test_predictions = pd.Series(test_predictions.reshape(test_predictions.shape[0],))
train_predictions1 =pd.Series(train_predictions.reshape(train_predictions.shape[0],))
pred_df = pd.DataFrame(y_test,columns = ['Test TRUE Y'])
pred_df = pd.concat([pred_df,test_predictions],axis = 1)
pred_df.columns = ['Test true y', 'Pred']
train_df = pd.DataFrame(y_train,columns = ['Test TRUE Y'])
train_df = pd.concat([train_df,train_predictions1],axis = 1)
train_df.columns = ['Test true y', 'Pred']

#Wykres korelacji
sns.scatterplot(x = 'Test true y', y = 'Pred', data = train_df)
sns.scatterplot(x = 'Test true y', y = 'Pred', data = pred_df, alpha =0.2)

#Zapis zbiorów do CSV
train_df.to_csv('train_set_GESTOSC_GELU_ALPHA.csv', sep=';',encoding='utf-8')
pred_df.to_csv('test_set_GESTOSC_GELU_ALPHA.csv', sep=';',encoding='utf-8')

'''Zapis modelu (do późniejszego wykorzystania bez konieczności
trenowania sieci ponownie)'''

model.save("Model_Rho_ALPHA_G1.h5")


def predictions3(MC, MA, SYM, P, T, vb=0):
    '''
    Ta funkcja iteruje po ciśnieniu i temperaturze dla cieczy jonowej..

    Parametry:
    MC (float): Masa kationu.
    MA (float): Masa anionu.
    SYM (int): Sparametr IS_SYM (0,1).
    P (list): Lista wartości ciśnienia.
    T (list): Lista wartości temperatury.
    vb (int, optional): Poziom szczegółowości dla predykcji modelu. Domyślnie 0.

    Returns:
    list: List of model predictions for all combinations of pressure and temperature.'''

    res = [model.predict(scaler.transform([[MC, MA, SYM, i, j]]), verbose=vb)[0] for i in P for j in T]
    return res
'''Przykładowa ciecz jonowa'''
nazwa = 'C2ImC1OC6_NTF2'
Mcat = 211.181
Man = 280.146

P = [0.1019,9.81,19.62,29.43,39.24,49.05,58.86,68.67,78.48,88.29,98.1,107.91,117.72,127.53,137.34,147.15,156.96,166.77,176.58,186.39,196.2]
T = [293.75,312.85,333.15,352.95,373.25]
result = predictions3(Mcat,Man,0,P,T);

res_flat = np.array(result).flatten()  # przekształcenie do jednowymiarowej tablicy numpy
res_numerical = [val.item() for val in res_flat]  # wyodrębnieniewartości liczbowych

#Pętla do sprawdzenia „ad hoc”
# for element in res_numerical:
#     print(element)

#Ta część ma za zadanie przekonwertować tablice 1D na tablice o wymiarach zgodnych z danymi eksperymentalnymi
tablica_danych = np.array(res_numerical).reshape(len(P), len(T))
cisnienie = pd.DataFrame(P, columns=['P'])
nowe_naglowki = {'P': 'P'}
nowe_nazwy_kolumn = T
for i, temperature in enumerate(nowe_nazwy_kolumn):
    nowe_naglowki[i] = temperature
dane = pd.concat([cisnienie, pd.DataFrame(tablica_danych)], axis=1)
dane.rename(columns=nowe_naglowki, inplace=True)
dane.to_excel(nazwa+'_PURE_R_DATA.xlsx', index=False) ###Zapis do pliku Excel
