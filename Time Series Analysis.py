#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[2]:


df_airline= pd.read_csv("AirPassengers.csv")


# In[3]:


df_airline


# In[4]:


df_airline.isnull().sum()


# In[5]:



df_airline.info()


# In[6]:


df_airline['Month']=pd.to_datetime(df_airline['Month'])


# In[7]:


df_airline.set_index('Month',inplace=True)


# In[8]:


df_airline


# In[9]:


plt.figure(figsize=(15,500))
df_airline.plot() 
#seasonal-upward trend


# In[10]:


from statsmodels.tsa.stattools import adfuller


# In[11]:


def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p- value: {}'.format(result[1]))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[12]:


adf_test(df_airline['Passengers'])


# In[13]:


df_airline['Passengers First Difference']=df_airline['Passengers']-df_airline['Passengers'].shift(1)


# In[14]:


df_airline


# In[15]:


adf_test(df_airline['Passengers First Difference'].dropna())


# In[16]:


df_airline['Passengers second Difference']=df_airline['Passengers']-df_airline['Passengers'].shift(2)


# In[17]:


df_airline


# In[18]:


adf_test(df_airline['Passengers second Difference'].dropna())


# In[19]:


# here our d is define as stationarity is achived at diff. 2    d=2


# In[20]:



from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[21]:


acf = plot_acf(df_airline["Passengers"].dropna())  


# In[22]:


pacf = plot_pacf(df_airline["Passengers"].dropna())  


# In[23]:



acf = plot_acf(df_airline['Passengers First Difference'].dropna())  
pacf2 = plot_pacf(df_airline['Passengers First Difference'].dropna())


# In[24]:


acf2 = plot_acf(df_airline["Passengers second Difference"].dropna())##from acf plot our MA means q = 1 ,
                                                                     ##Acf plot is also idicates that seasnality is present 
pacf2 = plot_pacf(df_airline["Passengers second Difference"].dropna()) #p = 


# In[25]:



from datetime import datetime,timedelta
train_dataset_end=datetime(1955,12,1)
test_dataset_end=datetime(1960,12,1)


# In[26]:


train_data=df_airline[:train_dataset_end]
test_data=df_airline[train_dataset_end+timedelta(days=1):test_dataset_end]


# In[27]:



##prediction
pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]


# In[28]:


test_data


# In[29]:


test_data.tail()


# In[30]:


from statsmodels.tsa.arima_model import ARIMA


# In[31]:


train_data.head()


# In[32]:


train_data.tail()


# In[33]:


# model_ARIMA=ARIMA(train_data['Passengers'],order=(1,0,1))


# In[34]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[35]:


model_SARIMA=SARIMAX(train_data['Passengers'],order=(3,0,1),seasonal_order=(0,2,0,12))


# In[36]:


model_SARIMA_fit=model_SARIMA.fit()


# In[37]:


model_SARIMA_fit.summary()


# In[38]:


pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]
print(pred_start_date)
print(pred_end_date)


# In[39]:


pred_Sarima=model_SARIMA_fit.predict(start=datetime(1956,1,1),end=datetime(1960,12,1))
residuals=test_data['Passengers']-pred_Sarima


# In[40]:


residuals.sum()


# In[41]:


model_SARIMA_fit.resid.plot()


# In[42]:


model_SARIMA_fit.resid.plot(kind='kde')


# In[43]:



test_data['Predicted_SARIMA']=pred_Sarima


# In[44]:


test_data.head()


# In[45]:


test_data[['Passengers','Predicted_SARIMA']].plot()


# In[46]:


df_airline["Predicted"] = test_data['Predicted_SARIMA']


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


plt.figure(figsize=(25,15))
plt.plot(df_airline["Predicted"])
plt.plot(df_airline["Passengers"])
plt.legend(["Predicted SARIMA", "PASSENGERS"], loc ="lower right",fontsize="40")
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 25)
plt.show()


# In[49]:


from pmdarima.arima import auto_arima


# In[50]:


arima_model = auto_arima(train_data['Passengers'],seasonal=True,m=12)


# In[51]:


arima_model


# In[ ]:





# In[52]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[53]:


Trend = df_airline['Passengers'] * 2.75


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


plt.plot( df_airline['Passengers'], Trend, 'c.')
plt.title("Trend against Time")
plt.xlabel("Minutes")
plt.ylabel("product demand")
plt.show()


# In[56]:


seasonality = 10 + np.sin(df_airline['Passengers']) * 10


# In[57]:


plt.figure(figsize=(15,5))
plt.plot(seasonality)


# In[58]:


np.random.seed(10)  # for result reproducibility
residual = np.random.normal(loc=0.0, scale=1, size=len(df_airline['Passengers']))


# In[59]:


plt.plot(df_airline['Passengers'], residual, 'r-.')
plt.title("Residuals against Time")
plt.xlabel("Minutes")
plt.ylabel("Product demand");


# In[60]:


ignored_residual = np.ones_like(residual)
# we multiply other components to create a multiplicative time series
multiplicative_Tmodel = Trend * seasonality * ignored_residual


# In[61]:


plt.plot(df_airline['Passengers'], multiplicative_Tmodel, 'k-.')
plt.title("Multiplicative Time Series")
plt.xlabel("Minutes")
plt.ylabel("product demand");


# In[62]:


ts_decompose_add = seasonal_decompose(x = df_airline['Passengers'], 
                                          model="multiplicative") # the frequency of fluctuation is more than one year thus cyclic component
estimated_trend_add = ts_decompose_add.trend
estimated_seasonal_add = ts_decompose_add.seasonal
estimated_residual_add = ts_decompose_add.resid


# In[63]:


fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(15)

axes[0].plot(df_airline['Passengers'], label='Original')
axes[0].legend(loc='upper left');

axes[1].plot(estimated_trend_add, label='Trend')
axes[1].legend(loc='upper left');

axes[2].plot(estimated_seasonal_add, label='Cyclic')
axes[2].legend(loc='upper left');

axes[3].plot(estimated_residual_add, label='Residuals')
axes[3].legend(loc='upper left');


# In[64]:


ts_decompose_add = seasonal_decompose(x = df_airline['Passengers'], 
                                          model="additive") # the frequency of fluctuation is more than one year thus cyclic component
estimated_trend_add = ts_decompose_add.trend
estimated_seasonal_add = ts_decompose_add.seasonal
estimated_residual_add = ts_decompose_add.resid


# In[65]:


fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(15)

axes[0].plot(df_airline['Passengers'], label='Original')
axes[0].legend(loc='upper left');

axes[1].plot(estimated_trend_add, label='Trend')
axes[1].legend(loc='upper left');

axes[2].plot(estimated_seasonal_add, label='Cyclic')
axes[2].legend(loc='upper left');

axes[3].plot(estimated_residual_add, label='Residuals')
axes[3].legend(loc='upper left');


# In[ ]:





# # Holts winter exponential smoothing

# In[66]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[67]:


hwmodel = ExponentialSmoothing(train_data["Passengers"],trend="add",seasonal="mul",seasonal_periods=12)


# In[68]:


model_HWmodel_fit=hwmodel.fit()


# In[69]:


pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]
print(pred_start_date)
print(pred_end_date)


# In[ ]:





# In[70]:


testpred = model_HWmodel_fit.forecast(70)


# In[71]:


model_HWmodel_fit.resid.sum()


# In[72]:


model_HWmodel_fit.resid.plot(kind='kde')


# In[73]:


df_airline["Predicted_holtswinter"] = testpred


# In[74]:


plt.figure(figsize=(30,15))
plt.plot(df_airline["Predicted"])
plt.plot(df_airline["Passengers"])
plt.plot(df_airline["Predicted_holtswinter"])
plt.legend(["Predicted by SARIMA", "PASSENGERS","Predicted by holtwinter"], loc ="lower right",fontsize="40")
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 25)
plt.show()


# In[75]:


residuals=test_data['Passengers']-testpred


# In[76]:


residual.sum()


# In[77]:


model_HWmodel_fit.resid.plot()


# # LSTM   
# ##### https://github.com/krishnaik06/Stock-MArket-Forecasting/blob/master/Untitled.ipynb

# In[78]:


df_airline


# In[79]:


from datetime import datetime,timedelta
train_dataset_end=datetime(1955,12,1)
test_dataset_end=datetime(1960,12,1)


# In[80]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df_airline["Passengers"]).reshape(-1,1))


# In[81]:


df1


# In[82]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[83]:


training_size,test_size


# In[84]:



train_data


# In[85]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[86]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[87]:



print(X_train.shape), print(y_train.shape)


# In[88]:


print(X_test.shape), print(ytest.shape)


# In[89]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[90]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,C


# In[91]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=[['accuracy','mse']],)


# In[92]:


model.summary()


# In[93]:


model.summary()


# In[94]:


history = model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=500,batch_size=50,verbose=1)


# In[121]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim([0,0.03])
plt.xlim([0,30])


# In[96]:


import tensorflow as tf


# In[97]:


tf.__version__


# In[ ]:





# In[98]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[99]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[124]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(ytest,test_predict)))
print('Validation rmse:', np.sqrt(mean_squared_error(ytest, test_predict)))


# In[101]:


### Plotting 
# shift train predictions for plotting
look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 25)

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(["Passengers", "Predicted by traing data","predicted by test data"], loc ="lower right",fontsize="20")

plt.gcf().autofmt_xdate()
plt.show()


# In[123]:



mlp_train_pred = model.predict(X_train.values)
mlp_valid_pred = model.predict(X_valid.values)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, mlp_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, mlp_valid_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




