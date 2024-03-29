import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers
from sklearn.model_selection import train_test_split
df=pd.read_csv('DryBean_Class.csv')
X=df[['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','EquivDiameter','Extent','Solidity',	
'roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']]
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = models.Sequential()  #使用Sequential進行類神經網路堆疊
model.add(layers.Dense(64,activation='relu')) #輸入層(也屬隱藏層)
model.add(layers.Dense(64,activation='relu')) #隱藏層
model.add(layers.Dense(7,activation='softmax')) #輸出層
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history=model.fit(X_train,
                  y_train,
                  epochs=20,
                  batch_size=128,)
#model.summary()
answer_X = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
print('------Macro------')
print('Macro precision', precision_score(y_test, answer_X, average='macro'))
print('Macro recall', recall_score(y_test, answer_X, average='macro'))
print('Macro f1-score', f1_score(y_test, answer_X, average='macro'))
#Num
import pandas as pd
import numpy as np
df=pd.read_csv('Dry_Bean_Num.csv')
df.columns=[['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter',
'Extent','Solidity','roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4','Class']]
X=df[['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','EquivDiameter','Extent','Solidity',	
'roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4','Class']]
y=df['ConvexArea']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = models.Sequential()  
model.add(layers.Dense(64,activation='relu')) 
model.add(layers.Dense(32,activation='relu')) 
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['mse'])
history=model.fit(X_train,
                  y_train,
                  epochs=13,
                  batch_size=126,)

answer_X = model.predict(X_test)
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
print('-----MAE----RMSE----MAPE-----')
#MAE
from sklearn.metrics import mean_absolute_error
print('MAE:',mean_absolute_error(y_test,answer_X))
#RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(np.array(answer_X), np.array(y_test))
print("RMSE:" ,rmse_val)
#MAPE
from sklearn.metrics import mean_absolute_percentage_error
print('MAPE:',mean_absolute_percentage_error(y_test,answer_X))
