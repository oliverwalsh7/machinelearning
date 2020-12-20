# sentdex python machine learning tutorial

import pandas as pd
import quandl, math 
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace = True)

forecast_out = int(math.ceil(0.02*len(df))) # .1 = 10 days, .01 = 1 day
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) # shift adj close 1 day forward

df.dropna(inplace = True)

x = np.array(df.drop(['label'], 1)) # features
y = np.array(df['label'])

x = preprocessing.scale(x) # normalize data
df.dropna(inplace = True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2) # 20 pct of test data

clf = LinearRegression() # regression algorithm
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test) # confidence

print (accuracy)