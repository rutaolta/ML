import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance

data = pd.read_csv('stars.csv', delimiter=',')
print(data)

X = data.drop('TARGET', axis=1)
y = pd.DataFrame(data['TARGET'])

print(round(X['MIP'].mean(), 3))
x_norm = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(x_norm, columns=X.columns)
print(round(X['MIP'].mean(), 3))

reg = LogisticRegression(random_state=12, solver='lbfgs').fit(X, y.values.ravel())
new_star = np.array([0.134, 0.085, 0.831, 0.214, 0.934, 0.439, 0.597, 0.624])
other_stars = np.array(X)

print(reg.predict_proba([new_star]))

distances = []
for star in other_stars:
    distances.append(distance.euclidean(star, new_star))

data_dist = pd.DataFrame(X)
data_dist['TARGET'] = y
data_dist['DISTANCE'] = distances
print(data_dist[data_dist['DISTANCE'] == data_dist['DISTANCE'].min()])
#print(data_dist[1:21])
p = data_dist.sort_values('DISTANCE')
#print(np.sort(data_dist))
print(p[1:21])