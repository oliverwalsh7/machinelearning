# sentdex python machine learning tutorial

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace = True)

forecast_out = int(math.ceil(0.01*len(df))) # .1 = 10 days, .01 = 1 day

df['label'] = df[forecast_col].shift(-forecast_out) # shift adj close 1 day forward

x = np.array(df.drop(['label'], 1)) # features
y = np.array(df['label'])

x = preprocessing.scale(x) # normalize data
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2) # 20 pct of test data

clf = LinearRegression(n_jobs = -1) # regression algorithm
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test) # confidence

forecast_set = clf.predict(x_lately)

print (forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix_date = last_date.timestamp()
one_day = 86400
next_unix_day = last_unix_date + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix_day)
    next_unix_day += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()