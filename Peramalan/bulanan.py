import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import csv, json
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from pandas.tseries.offsets import DateOffset


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

#warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)

# Import data
col_names = ['id_peramalan','hasil_ramal']

df= pd.read_csv('Data-Tahunan-v3.csv',index_col=0, delimiter=";")

# ADF Test
def adf_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print("1. ADF: ", dftest[0])
    print("2. P-Value: ", dftest[1])
    print("3. Num of Lags: ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression: ", dftest[3])
    print("5. Critical Values: ")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)
        
adf_test(df.value)

# Mencari model terbaik
# stepwise_fit = auto_arima(df.value, 
#                           d=1, 
#                           trace=True, 
#                           m=12,
#                           seasonal=True,
#                           suppress_warnings=True)

# print(df.shape)
# print(df.value)

# 0,1,1 ARIMA Model
model = SARIMAX(df['hasil_ramal'], order=(0,1,1),seasonal_order=(1, 0, 0, 12))
model_fit = model.fit(disp=0)

train = df.iloc[:-30]
test = df.iloc[-30:]

start = len(train)
end = len(train) + len(test)-1

prediction = pd.DataFrame(model_fit.predict(start=start, end=end, typ='levels'), index=test.index)
prediction.columns = ['hasil_prediksi']

#print(prediction)

data = df['hasil_ramal']

start_predict = end+1
end_predict = start_predict+12

list_indeks = list()

for i in range(12):
    list_indeks.append(i+start_predict+1)

forecast = pd.DataFrame(model_fit.predict(start=start_predict, end=end_predict, typ='levels'), index=list_indeks)

forecast.columns = ['hasil_ramal']
forecast.index.rename('bulan_ke', inplace=True)

# data=pd.concat([data,forecast])
print(forecast)

# print(data)

# forecast.to_csv('hasilprediksi.csv')
jsonfile =  forecast.to_json(orient='table')
print(jsonfile)

# Serializing json 
json_object = json.dumps(jsonfile, indent=4, separators=(',', ': '),sort_keys=True)
print(json_object)
  
# Writing to sample.json
with open("hasilbulanan.json", "w") as outfile:
    outfile.write(jsonfile)

plt.figure(figsize=(8,5))
plt.plot(train, label="Training")
plt.plot(test, label="Test")
plt.plot(prediction, label="Predicted")
plt.plot(forecast, label="Forecast")
plt.legend(loc='upper left')
# plt.savefig('Prediction-0-1-1.jpg')
plt.show()