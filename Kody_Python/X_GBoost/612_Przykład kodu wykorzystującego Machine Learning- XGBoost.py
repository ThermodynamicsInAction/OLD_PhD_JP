#Import bibliotek – początkowa struktura jest analogiczna jak w
#przypadku sieci neuronowej.

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb #Załadowanie biblioteki xgboost
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('merged_data1.csv',encoding = 'utf8',sep=';')
X = df[['M_C', 'M_A', 'IS_SYM', 'P', 'T']].values
y = df[['Rho']].values

'''Przygotowanie danych - podzielenie danych na zbiór trenujacy i testowy'''
X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size = 0.4, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print('Test shapes')
print(X_test.shape)
print(y_test.shape)

# Skalowanie cech dla lepszej wydajności modelu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.ravel()
y_test = y_test.ravel()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # Typ zadania: regresja
    'max_depth': 4,                  # Maksymalna głębokość drzewa
    'learning_rate': 0.1,
    #'num_boost_rounds': 100,# Współczynnik uczeni             # Liczba drzew w modelu
    'random_state': 42               # Ziarno losowości dla reprodukowalności wyników
}
xgb_model = xgb.train(params, dtrain, num_boost_round=300)

'''Model training'''
y_test_xgb = xgb_model.predict(dtest)
y_train_xgb = xgb_model.predict(dtrain)

### Metryki dla zbioru testowego
r2 = r2_score(y_test, y_test_xgb)
print(f"Współczynnik determinacji R^2: {r2}")
mse = mean_squared_error(y_test, y_test_xgb)
print(f"Błąd średniokwadratowy MSE: {mse}")

###Metryki dla zbioru treningowego
r2_train = r2_score(y_train, y_train_xgb)
print(f"Współczynnik determinacji R^2: {r2_train}")
mse_train = mean_squared_error(y_train, y_train_xgb)
print(f"Błąd średniokwadratowy MSE: {mse_train}")

# Przeprowadź walidację krzyżową z 25 foldami i oblicz R^2 jakość modelu
cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_train, y_train, cv=25, scoring='r2')
# Oblicz średnią wartość R^2 po wszystkich foldach
mean_r2 = cv_scores.mean()
print("Średni R^2 po walidacji krzyżowej:", mean_r2)

cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_test, y_test, cv=25, scoring='r2')
# Oblicz średnią wartość R^2 po wszystkich foldach
mean_r2 = cv_scores.mean()
print("Średni R^2 po walidacji krzyżowej:", mean_r2)


'''Dodane komentarze
Oblicz predykcje dla wartości testowych'''
test_predictions_xgb = pd.Series(y_test_xgb.reshape(1275,))
pred_df = pd.DataFrame(y_test,columns = ['Test TRUE Y'])
pred_df = pd.concat([pred_df,test_predictions_xgb],axis = 1)
pred_df.columns = ['Test true y', 'Pred']

'''Predykcje dla wartości treningowych'''
train_predictions_xgb = pd.Series(y_train_xgb.reshape(X_train.shape[0],))
train_df = pd.DataFrame(y_train,columns = ['Test TRUE Y'])
train_df = pd.concat([train_df,train_predictions_xgb],axis = 1)
train_df.columns = ['Test true y', 'Pred']


'''Analogicznie jak w przypadku sieci neuronowej zaimplementowano wykres- na listingu w dodatku 6.1.2 nie ma tego'''
sns.scatterplot(x = 'Test true y', y = 'Pred', data = train_df)
sns.scatterplot(x = 'Test true y', y = 'Pred', data = pred_df, alpha = 0.2)

import os
directory_path = 'res'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
train_df.to_csv(os.path.join(directory_path, 'train_set_XB_R.csv'), sep=';', encoding='utf-8')
pred_df.to_csv(os.path.join(directory_path, 'test_set_XB_R.csv'), sep=';', encoding='utf-8')

'''Blok pozwalający na predykcje- analogiczny jak w NN'''
def predictions3(MC,MA,SYM,P,T):
    res = []
    for j in T:
        for i in P:
            new_geom = [[MC,MA,SYM,i,j]]
            new_geom = scaler.transform(new_geom)
            new_geom_dmatrix = xgb.DMatrix(new_geom)  # Konwersja na DMatrix
            res.append(xgb_model.predict(new_geom_dmatrix))
            #print(model.predict(new_geom))
    return res
'''Przykład'''
nazwa = 'C4Mim_C1SO4'
Mcat = 139.29
Man = 111.097

P = [0.1,2.5,5,10,15,20,25,30,35]
T = [283.15,285.65,288.15,290.65,293.15,295.65,298.15,300.65,303.15,305.65,308.15,310.65,313.15,315.65,318.15,320.65,323.15,325.65,328.15,330.65,333.15,335.65,338.15,340.65,343.15,345.65,348.15,350.65,353.15]
result = predictions3(Mcat,Man,0,P,T);

'''Zapis wyniku do tabeli o wymiarze Cisnienie x Temperatura'''
res_flat = np.array(result).flatten()  # przekształcenie do jednowymiarowej tablicy numpy
res_numerical = [val.item() for val in res_flat]  # wyodrębnienie wartości liczbowych
directory_path = nazwa
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
tablica_danych = np.array(res_numerical).reshape(len(T), len(P))
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
dane = pd.DataFrame(tablica_danych)
dane = dane.T
dane.to_excel(directory_path+nazwa+'_Rho_DATA_XB.xlsx', index=False)
raw = pd.DataFrame(res_numerical) ###Zapis wektora danych w postaci kolumny surowych danych
raw.to_excel(directory_path+nazwa+'_Rho_RAW_XB.xlsx', index=False)

try:
    dane.to_excel(os.path.join(directory_path, nazwa+'_Rho_DATA_XGB.xlsx'), index=False)
    raw = pd.DataFrame(res_numerical)
    raw.to_excel(os.path.join(directory_path, nazwa+'_Rho_RAW_XGB.xlsx'), index=False)
    print("Pliki zostały zapisane poprawnie.")
except Exception as e:
    print("Wystąpił błąd podczas zapisywania plików:", e)