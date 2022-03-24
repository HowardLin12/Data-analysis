import pandas as pd
import numpy as np
df=pd.read_csv('DryBean_Class.csv')

X=df[['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','EquivDiameter','Extent','Solidity',	
'roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']]
y=df['Class']

from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers

from sklearn.model_selection import train_test_split
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