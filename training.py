import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

global df
df = pd.read_csv('carsTDIA.csv', sep=';')

def convertmanufacturer_name(manufacturer:str, columndataframe):
    index= df['manufacturer_name'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]

def convertmodel_name(manufacturer:str, columndataframe):
    index= df['model_name'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]

def converttransmission(manufacturer:str, columndataframe):
    index= df['transmission'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]

def convertcolor(manufacturer:str, columndataframe):
    index= df['color'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]

def convertengine_fuel(manufacturer:str, columndataframe):
    index= df['engine_fuel'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]

def convertengine_type(manufacturer:str, columndataframe):
    index= df['engine_type'].loc[lambda x: x==manufacturer].index[0]
    return columndataframe[index]


def predictannouce(manufacturer_name:str, model_name:str, transmision:str, color:str, odometer_value:int, year:int, engine_fuel:str, idengine_type:str, price:float):
    le = LabelEncoder()

    # extraction des colonnes une par une
    X0 = df.iloc[:,0]
    X1 = df.iloc[:,1]
    X2 = df.iloc[:,2]
    X3 = df.iloc[:,3]
    X4 = df.iloc[:,4]
    X5 = df.iloc[:,5]
    X6 = df.iloc[:,6]
    X7 = df.iloc[:,7]
    X8 = df.iloc[:,8]

    # conversion en int pour les données en string
    X0 = le.fit_transform(X0)
    X1 = le.fit_transform(X1)
    X2 = le.fit_transform(X2)
    X3 = le.fit_transform(X3)
    X6 = le.fit_transform(X6)
    X7 = le.fit_transform(X7)

    # conversion no
    idmanufacturer_name= convertmanufacturer_name(manufacturer_name, X0)
    idmodel_name= convertmodel_name(model_name, X1)
    idtransmision= converttransmission(transmision, X2)
    idcolor= convertcolor(color, X3)
    idengine_fuel= convertengine_fuel(engine_fuel, X6)
    idengine_type= convertengine_type(idengine_type, X7)

    # concaténation de toutes les colonnes
    X= np.column_stack((X0, X1, X2, X4, X5, X3, X6, X7))
    # #sauf le prix qui sert à trust l'annonce
    Y = X8

    # 80% des données pour l'entrainement, le reste pour les tests
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    rf = ExtraTreesRegressor()
    #entrainement
    rf.fit(X_train, Y_train)

    # tentative de prédiction avec le modèle entrainé
    Y_rf_train_pred = rf.predict(X_train)
    # #pareil avec le test
    Y_rf_test_pred = rf.predict(X_test)

    plt.figure(figsize=(5,5))
    plt.scatter(x=Y_train, y=Y_rf_train_pred, c="#7CAE00", alpha=0.3)
    z = np.polyfit(Y_train, Y_rf_train_pred, 1)
    #polynomiale
    p = np.poly1d(z)
    plt.plot(Y_train,p(Y_train),"#F8766D")
    #affichage d'une version graphique des prédictions
    plt.show()
    testdata = [idmanufacturer_name,idmodel_name,idtransmision, odometer_value, year, idcolor, idengine_fuel, idengine_type] #fausse données envoyé par l'utilisateur (déjà converties en int)

    # n'accépte pas les tableau en une dimension, reformatage
    test = rf.predict(np.array(testdata).reshape((1, -1)))

    # calcul proba par écart à la moyenne
    maxval = max(test, price)
    minval = min(test, price)
    return float((1-(maxval-minval)/(maxval)))


#print(float(predictannouce("Subaru", "Outback", "automatic", "silver", 190000, 2010, "gasoline","gasoline",10900.0)))