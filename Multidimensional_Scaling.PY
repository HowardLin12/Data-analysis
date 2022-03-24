import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import pairwise_distances

x=np.array([[0,36.381,66.275,159.829,212.576,307.856,339.284],
            [36.381,0,29.894,123.448,176.195,271.575,302.903],
            [66.275,29.894, 0,93.554,146.301,241.681,273.009],
            [159.829,123.448,93.554,0,52.747,148.127,179.455],
            [212.576,176.195,146.301,52.747,0,95.38,126.708],
            [307.856,271.575,241.681,148.127,95.38,0,31.328],
            [339.284,302.903,273.009,179.455,126.708,31.328,0]])

#print(x.dtype)
from sklearn.preprocessing import MinMaxScaler
minimax_x=MinMaxScaler()
x=pd.DataFrame(minimax_x.fit_transform(x))

dis_matrix=pairwise_distances(x,metric='euclidean')
#print(dis_matrix.shape)
mds_model=manifold.MDS(n_components=2,dissimilarity="precomputed")
mds_fit=mds_model.fit(dis_matrix)
mds_coords=mds_model.fit_transform(dis_matrix)
city_names=['Taipei','Taoyuan','Hsinchu','Taichung','Yunlin','Tainan','Kaohsiung']
#print(city_names)
plt.figure()
plt.scatter(mds_coords[:,0],mds_coords[:,1],
    facecolors = 'none', edgecolors = 'r')  
labels = city_names
for label, x, y in zip(labels, mds_coords[:,0], mds_coords[:,1]):
    plt.annotate(label, (x,y), xycoords = 'data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('City')    
plt.show()

