# We'll be using Sklearn library for this task..
from sklearn.linear_model import LinearRegression 
# pandas and numpy are used for data manipulation 
import pandas as pd 
import numpy as np 
# matplotlib and seaborn are used for plotting graphs 
import matplotlib.pyplot as plt 
import seaborn 
# fix_yahoo_finance is used to fetch data 
import  yfinance    as yf
#https://medium.com/analytics-vidhya/using-linear-regression-to-predict-aapl-apple-stock-prices-in-python-1a629fbea15b


# Read data
xx='t'
df = yf.download(xx, start='2022-01-01', end=(pd.to_datetime('today')), progress=False, auto_adjust = True)
df.reset_index(inplace=True)
df['u']='yahoo raw'
print(df,' from yahoo raw')
df.drop('u',axis=1,inplace=True)
# Only keep close columns 
df=df[['Close']] 
# Drop rows with missing values 
df= df.dropna() 
# Plot the closing price of GLD 
df.Close.plot(figsize=(10,5)) 
plt.ylabel("AAPL Prices")
plt.show()

df['S_3'] = df['Close'].shift(1).rolling(window=3).mean() 
df['S_9']= df['Close'].shift(1).rolling(window=9).mean() 
df= df.dropna() 
X = df[['S_3','S_9']] 
X.head()
y = df['Close']
y.head()

t=.8 
t = int(t*len(df)) 
# Train dataset 
X_train = X[:t] 
y_train = y[:t]  
# Test dataset 
X_test = X[t:] 
y_test = y[t:]

##Y = m1 * X1 + m2 * X2 + C AAPL price = m1 * 3 days moving average + m2 * 15 days moving average + c

linear = LinearRegression().fit(X_train,y_train)

predicted_price = linear.predict(X_test)  
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
predicted_price.plot(figsize=(10,5))  
y_test.plot()  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("AAPL Price")  
plt.show()
