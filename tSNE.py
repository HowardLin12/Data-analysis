import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import codecs
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder 
#Import DATA
labelencoder = LabelEncoder() 
df =pd.read_csv('1ofk.csv',encoding='utf-8')
df1=pd.read_csv('sim2.csv',encoding='utf-8')
df2=pd.read_csv('Drink Random Data.csv',encoding='utf-8')
y  =pd.read_csv('tARGET.csv',encoding='utf-8')
y  =labelencoder.fit_transform(y)
y=np.array(y)
dfone=pd.get_dummies(df)
dfcom=np.hstack((dfone,df1,df2))
dftotal=np.array(dfcom)
#TSNE
mds_model=manifold.TSNE(n_components=2,random_state=None,verbose=0,perplexity=80,n_iter=250,init='random',angle=0.5)
x=dfcom
mds_fit=mds_model.fit(x)
mds_coords=mds_model.fit_transform(x)
mds_coords=np.array(mds_coords)
#Hierarchical 
from sklearn.cluster import AgglomerativeClustering
ml=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='ward')
ml.fit_predict(mds_coords)
df["y"]=y
df['Dimention-1']=mds_coords[:,0]  #mds_final.iloc
df['Dimention-2']=mds_coords[:,1]
import seaborn as sns
sns.scatterplot(x="Dimention-1", y="Dimention-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 7),
                data=df).set(title="Drink-simularity") 
plt.show()