from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
X, Y = [], []
X.append('a')
X.append('b')
X.append('c')
X.append('d')
X.append('f')

X.append('e')
X.append('e')
X.append('e')
X.append('e')
X.append('e')

Y.append(0)
Y.append(0)
Y.append(0)
Y.append(0)
Y.append(0)

Y.append(1)
Y.append(1)
Y.append(1)
Y.append(1)
Y.append(1)

all_data = pd.DataFrame({'data': X, 'label': Y})
train, val = train_test_split(all_data, test_size=0.2)
print(train)
print(val)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
X = np.array(X)
Y = np.array(Y)
for train_index, val_index in skf.split(X, Y):
    trainX, valX = X[train_index], X[val_index]
    trainY, valY = Y[train_index], Y[val_index]
    print(trainX, trainY)
    print('------------')
    print(valX, valY)
    
