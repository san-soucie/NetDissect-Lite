import statsmodels.api as sm
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import *

# rc('text', usetex=True)

dfo = pd.read_csv('brainscore-netdissect-correlations-revised2.csv')
df = dfo
# df = df.dropna()
# print(df)

X4 = df[['Concept Ratio']]
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

X = X4
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

# test = dfo[['NetDissect Ratio', 'NetDissect Concepts']][-2:]
# test = dfo[['V4', 'IT', 'Behaviour', 'Imagenet']][-2:]
# test = sm.add_constant(test)
# print(model.predict(test))

x = 'NetDissect Concepts'
y = 'IT'

for x in ['NetDissect Units', 'NetDissect Ratio', 'NetDissect Concepts', 'Imagenet']:
    for y in ['Behaviour', 'IT']:
        fig = df.plot.scatter(x=x, y=y)
        dx, dy = .01 * (fig.get_xlim()[1] - fig.get_xlim()[0]), 0.01 * (fig.get_ylim()[1] - fig.get_ylim()[0])
        for i, point in df.iterrows():
            if 'cornet' in str(point['Model']):  # or 'alexnet' in str(point['Model']):
                fig.text(point[x] + dx, point[y] + dy, str(point['Model'].split()[0]))
        data_x = df[x]
        data_y = df[y]
        spearman = st.spearmanr(data_x[~np.isnan(data_x)], data_y[~np.isnan(data_x)])
        # plt.title(r'\Big{' + f'{x} vs {y}' + r'}' + r'\newline\tiny{Spearman Correlation: '
        #          + f'{spearman[0]:.3f} (p={spearman[1]:.3f})' + r'}')
        plt.title(f'{x} vs {y} Score\nSpearman Correlation: {spearman[0]:.3f} (p={spearman[1]:.3f})')
        xname = x.replace(' ', '_')
        yname = y.replace(' ', '_')
        # plt.savefig(xname + '_' + yname + '.png')
        plt.show()

import pretrainedmodels.models as models