import statsmodels.api as sm
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib

dfo = pd.read_csv('brainscore-netdissect-correlations-revised2-normalized.csv')
df = dfo
#df = df.dropna()
#print(df)

X3 = df[['NetDissect Concepts']]
X2 = df[['NetDissect Ratio']]
X1 = df[['NetDissect Units']]
XND = df[['NetDissect Ratio', 'NetDissect Concepts']]

y1 = df['IT']
y2 = df['Behaviour']
y0 = df['Brain-Score']
y3 = df['V4']
y4 = df['Imagenet']
Xbs = df[['V4', 'IT', 'Behaviour', 'Imagenet']]

X = XND
y = y2

Xc = X
Xc = sm.add_constant(Xc)

model = sm.OLS(y, Xc, missing='drop').fit()
predictions = model.predict(Xc)
print(model.summary())

model = sm.RLM(y, Xc, missing='drop').fit()
predictions = model.predict(Xc)
print(model.summary())

y = y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]
print(st.spearmanr(X, y)[0])
print(st.spearmanr(X, y)[1])

test = dfo[['NetDissect Ratio', 'NetDissect Concepts']][-2:]
#test = dfo[['V4', 'IT', 'Behaviour', 'Imagenet']][-2:]
test = sm.add_constant(test)
print(model.predict(test))