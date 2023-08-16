import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
x=iris.data
y=iris.target

x = x[y<2]
y = y[y<2]
x = x[:, :2]

index_list= np.arange(len(x))

np.random.shuffle(index_list)
bdry_idx = int(0.7*(len(index_list))) #GA, SGA할 때는 0.8

tr_x = x[index_list[:bdry_idx]]
val_x = x[index_list[bdry_idx:]]
tr_y = y[index_list[:bdry_idx]]
val_y = y[index_list[bdry_idx:]]





