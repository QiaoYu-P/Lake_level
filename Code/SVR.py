import os
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.svm import SVR
import warnings
from prettytable import PrettyTable
warnings.filterwarnings("ignore")
dataset=pd.read_excel("Test result.xlsx")
print(dataset)

values = dataset.values[:,2:]
values = np.array(values)
num_samples = values.shape[0]
per=np.arange(num_samples)
n_train_number = per[:int(num_samples * 0.8)]
n_test_number = per[int(num_samples * 0.8):]
Xtrain = values[n_train_number, :-1]
Ytrain = values[n_train_number, -1]
Ytrain = Ytrain.reshape(-1,1)
Xtest = values[n_test_number, :-1]
Ytest = values[n_test_number,  -1]
Ytest = Ytest.reshape(-1,1)

m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)
m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)

model = SVR(C=15, epsilon=0.01, gamma='auto')
model.fit(vp_train, vt_train)
yhat = model.predict(vp_test)
yhat = yhat.reshape(-1, 1)

predicted_data = m_out.inverse_transform(yhat)

print(predicted_data)
df=pd.DataFrame(predicted_data, columns=['Test result'])
df.to_excel(os.path.join('result',"excel.xlsx"))

def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        record.append(temp_mape)
    return np.mean(record) * 100

def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    table = PrettyTable(['Test set pointer','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        if n_out == 1:
            strr = 'Prediction outcome index：'
        else:
            strr = 'No.'+ str(i + 1)
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])
    return mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table

mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, 1)

print(table)

from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False
         }
rcParams.update(config)
plt.ion()


plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 2), dpi=300)
x = range(1, len(predicted_data) + 1)
plt.tick_params(labelsize=5)
plt.plot(x, predicted_data, linestyle="--",linewidth=0.8, label='predict',marker = "o",markersize=2)
plt.plot(x, Ytest, linestyle="-", linewidth=0.5,label='Real',marker = "x",markersize=2)
plt.rcParams.update({'font.size': 5})
plt.legend(loc='upper right', frameon=False)
plt.xlabel("Sample points", fontsize=5)
plt.ylabel("value", fontsize=5)
plt.title(f"The prediction result of bagging :\nMAPE: {mape(Ytest, predicted_data)} %",fontsize=5)

plt.ioff()
plt.show()