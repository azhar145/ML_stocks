#import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import  yfinance    as yf
from datetime import datetime, timedelta,date
import warnings
warnings.filterwarnings("ignore")

#to plot within notebook
import matplotlib.pyplot as plt
xx='spy'
no_of_days_in_future_to_predict=215
#read the data file
##df = pd.read_csv('D:\\python3\\data\\SensexHistoricalData.csv')
df = yf.download(xx, start='2022-01-01', end=(pd.to_datetime('today')), progress=False, auto_adjust = True)
df.reset_index(inplace=True)
df['u']='yahoo raw'
print(df,' from yahoo raw')
df.drop('u',axis=1,inplace=True)

#setting index as date
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']
df['status']='Actual'
#converting dates into number of days as dates cannot be passed directly to any regression model
df.index = (df.index - pd.to_datetime('1970-01-01')).days
print(df.shape)
# Convert the pandas series into numpy array, we need to further massage it before sending it to regression model
y = np.asarray(df['Close'])
x = np.asarray(df.index.values)

y2=np.asarray(df['High'])
y3=np.asarray(df['Low'])
y4=np.asarray(df['Open'])
y5=np.asarray(df['Volume'])

# Model initialization
# by default the degree of the equation is 1.
# Hence the mathematical model equation is y = mx + c, which is an equation of a line.
regression_model = LinearRegression()

regression_model_y2 = LinearRegression()
regression_model_y3 = LinearRegression()
regression_model_y4 = LinearRegression()
regression_model_y5 = LinearRegression()

# Fit the data(train the model)
regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

regression_model_y2.fit(x.reshape(-1, 1), y2.reshape(-1, 1))
regression_model_y3.fit(x.reshape(-1, 1), y3.reshape(-1, 1))
regression_model_y4.fit(x.reshape(-1, 1), y4.reshape(-1, 1))
regression_model_y5.fit(x.reshape(-1, 1), y5.reshape(-1, 1))

# Prediction for historical dates. Let's call it learned values.
y_learned = regression_model.predict(x.reshape(-1, 1))
y2_learned = regression_model_y2.predict(x.reshape(-1, 1))
y3_learned = regression_model_y3.predict(x.reshape(-1, 1))
y4_learned = regression_model_y4.predict(x.reshape(-1, 1))
y5_learned = regression_model_y5.predict(x.reshape(-1, 1))
print(df.shape,'   ',len(y_learned))




# Now, add future dates to the date index and pass that index to the regression model for future prediction.
# As we have converted date index into a range index, hence, here we just need to add 3650 days ( roughly 10 yrs)
# to the previous index. x[-1] gives the last value of the series.


newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + no_of_days_in_future_to_predict))
##print(df,'   ',df.shape)
##print('\n\n')
##print(df.values.reshape(-1,1),' reshape')



# Prediction for future dates. Let's call it predicted values.
y_predict = regression_model.predict(newindex.reshape(-1, 1))
y2_predict = regression_model_y2.predict(newindex.reshape(-1, 1))
y3_predict = regression_model_y3.predict(newindex.reshape(-1, 1))
y4_predict = regression_model_y4.predict(newindex.reshape(-1, 1))
y5_predict = regression_model_y5.predict(newindex.reshape(-1, 1))
print('\n\n\n')
print(' ***************************************************************************** ')
print('y_predict  44444 y_predict',len(y_predict),'     ',df.shape)
print('\n\n\n')
print('x   ',len(x))
print(' ***************************************************************************** ')
print('\n\n\n')
#print the last predicted value
print ("Closing price at 2029 would be around ", y_predict[-1])
##dq=pd.DataFrame([y_predict,y2_predict,y3_predict,y4_predict,y5_predict]).T
dq=pd.DataFrame(np.column_stack([y_predict,y2_predict,y3_predict,y4_predict,y5_predict]),columns=['Close', 'High', 'Low','Open','Volume'])
##dq=pd.DataFrane({'Close': y_predict,'High': y2_predict,'Low': y3_predict,'Open':y4_predict,'Volume': Volume            })

dq=dq[['Open','High','Low','Close','Volume']]
dq['status']='Predicted'
dq.index.names=['mm']
df.index.names=['mm']

ff=df['Date'].tail(1).values
df['Date'] = pd.to_datetime(df.Date)
##df.index = df['Date']
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']
from datetime import datetime as dt

dq['Date']=''
k=1
for x in dq.index:
    dq['Date'].loc[x]=pd.to_datetime(df['Date'][-1]+timedelta(x+1)).date()
    dq['Volume'].loc[x]=int(dq['Volume'].loc[x])
    k=k+1
####    df['Datep'].loc[x] = pd.to_datetime(df['Datep'].loc[x]).dt.

##dq['Datepx']=pd.to_datetime(df['Datep']).dt.date


dq=dq[['Date','Open','High','Low','Close','Volume','status']]
##df.drop_index(inplace=True)
##dq.drop_index(inplace=True)
dq.reset_index(drop=True,inplace=True)
dq.set_index('Date',inplace=True)
df.reset_index(drop=True,inplace=True)
df.set_index('Date',inplace=True)



df.reset_index(inplace=True)
dq.reset_index(inplace=True)

## shifting index
print(df.index[-1])
print(dq.index[0])
m5=df.index[-1]+1
dq.index=dq.index+m5
print(dq.index[0])


print('\n\n')
print(df,'   df')
print('\n\n')
print(dq,'  dq predicted stocks')
print(pd.to_datetime(df['Date']).dt.date[0])
print(pd.to_datetime(dq['Date']).dt.date)
y = np.asarray(df['Close'])
x = np.asarray(df.index.values)
y2 = np.asarray(dq['Close'])
x2 = np.asarray(dq.index.values)

import matplotlib.pyplot as plt
#plot the actual data
fig = plt.figure('MyName')
plt.figure(figsize=(16,8))
plt.plot(x,df['Close'],color='g', label='Close Price History')
plt.plot(x2,dq['Close'],color='r', label='Close Predicted stock ')
fig.canvas.set_window_title('Stock Market Predictions')



plt.legend()
plt.show()

##dq.reset_index('Date',drop=True,inplace=True)
##df.reset_index('Date',drop=True,inplace=True)
##
##date.today()+timedelta(1)
##
##
##sys.exit()


#to plot within notebook
##import matplotlib.pyplot as plt
##xx='tsla'
###read the data file
##df = yf.download(xx, start='2010-01-01', end=(pd.to_datetime('today')), progress=False, auto_adjust = True)
##df.reset_index(inplace=True)
##print(df,' 88888')
###setting index as date
####df['Date'] = pd.to_datetime(df.Date)
####df.index = df['Date']
####
#####converting dates into number of days as dates cannot be passed directly to any regression model
####df.index = (df.index - pd.to_datetime('1970-01-01')).days
##no_of_days_in_future_to_predict=1




'''









###setting index as date
##df['Date'] = pd.to_datetime(df.Date)
##df.index = df['Date']

#converting dates into number of days as dates cannot be passed directly to any regression model
##df.index = (df.index - pd.to_datetime('1970-01-01')).days

# Convert the pandas series into numpy array, we need to further massage it before sending it to regression model
y = np.asarray(df['Close'])
x = np.asarray(df.index.values)
yy=np.asarray(dq['Close'])
xx = np.asarray(dq.index.values)

# Model initialization
# by default the degree of the equation is 1.
# Hence the mathematical model equation is y = mx + c, which is an equation of a line.
regression_model = LinearRegression()


# Fit the data(train the model)
regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))


# Prediction for historical dates. Let's call it learned values.
y_learned = regression_model.predict(x.reshape(-1, 1))


# Now, add future dates to the date index and pass that index to the regression model for future prediction.
# As we have converted date index into a range index, hence, here we just need to add 3650 days ( roughly 10 yrs)
# to the previous index. x[-1] gives the last value of the series.
newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + no_of_days_in_future_to_predict))

# Prediction for future dates. Let's call it predicted values.
y_predict = regression_model.predict(newindex.reshape(-1, 1))

#print the last predicted value
print ("Closing price at 2029 would be around ", y_predict[-1])

#convert the days index back to dates index for plotting the graph #################
##x = pd.to_datetime(df.index, origin='1970-01-01', unit='D')
##future_x = pd.to_datetime(newindex, origin='1970-01-01', unit='D')
x = pd.to_datetime(df.index)
future_x = pd.to_datetime(newindex)

##xx = pd.to_datetime(dq.index, origin='1970-01-01', unit='D')
##future_x2 = pd.to_datetime(newindex2, origin='1970-01-01', unit='D')

xx = pd.to_datetime(dq.index)
future_x2 = pd.to_datetime(newindex2)

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#plot the actual data
plt.figure(figsize=(16,8))
plt.plot(x,df['Close'],color='g', label='Close Price History')
plt.plot(xx,dq['Close'], color='r',label='Close Price History2 ****')

#plot the regression model
plt.plot(x,y_learned, color='r', label='Mathematical Model')
plt.plot(xx,y_learned2, color='r', label='Mathematical Model2 ****')

#plot the future predictions
plt.plot(future_x,y_predict, color='b', label='Future predictions')
plt.plot(future_x2,y_predict2, color='b', label='Future predictions2 ****')

plt.suptitle('Stock Market Predictions', fontsize=16)

fig = plt.gcf()
fig.canvas.set_window_title('Stock Market Predictions')

plt.legend()
plt.show()
'''
