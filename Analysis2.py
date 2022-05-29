import imp
import lightgbm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, svm
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
#minimax
dataset=pd.read_csv(r"D:/DL/minimax_v2.csv")
#minimax=preprocessing.MinMaxScaler()
#data_minimax=minimax.fit_transform(dataset)
#df2=pd.DataFrame(data_minimax)
#df2.to_csv("D:/DL/minimax.csv") #檔案輸出

#print(data_minimax)
df=pd.DataFrame(dataset)
features=df.drop(["8"],axis=1) #特徵
targets=df["8"] #目標
from sklearn import preprocessing
from sklearn import utils
lab=preprocessing.LabelEncoder()
y_train_label=lab.fit_transform(targets)
print(y_train_label.shape)

#print(features)
#y_train_label=lab.fit(y_train)
X_train,X_test,y_train,y_test=train_test_split(features,y_train_label,train_size=0.75) #資料分割
#SVR
#建立SVR
#svr_rbf = SVR(C=1e3, kernel='rbf', gamma='auto')
#svr_rbf.fit(X_train,y_train)
#svr_predict=svr_rbf.predict(X_test)
#rndf相關


###############特徵選取################
#rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
#rfc.fit(X_train,y_train)
#importance=rfc.feature_importances_
#indices = np.argsort(importance)[::-1]
#features = X_train.columns
#for f in range(X_train.shape[1]):
#    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
from lightgbm import LGBMClassifier
model=LGBMClassifier()
model.fit(X_train,y_train)
from lightgbm import plot_importance
plot_importance(model,max_num_features=2)
plt.show()



#模型評估
#from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#print('\nThe value of default measurement of rbf SVR is', svr_rbf.score(X_test, y_test))
#print('R-squared value of rbf SVR is', r2_score(y_test, svr_predict))
#print('The mean squared error of rbf SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(svr_predict)))
#print('The mean absolute error of rbf SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(svr_predict)))
#print(svr_rbf.intercept_) #迴歸模型的截距

#輸出圖
#plt.scatter(svr_predict, y_test, color= 'black', label= 'Data') #數據點
#plt.plot(svr_predict, svr_predict, color= 'red', label= 'RBF model') #迴歸線
#plt.xlabel('data')
#plt.ylabel('quality')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor


#XGBOOST
#xgrg=XGBRegressor()
#xgrg.fit(X_train,y_train)
#print("Regression",xgrg.score(X_test,y_test))
#y_pred = xgrg.predict(X_test) # Predictions
#y_true = y_test # True values
#MSE = mse(y_true, y_pred)
#RMSE = np.sqrt(MSE)
#R_squared = r2_score(y_true, y_pred)
#print('MSE',MSE)
#print('RMSE',RMSE)
#print('R_squared',R_squared)
#plt.scatter(y_pred, y_test, color= 'black', label= 'Data') #數據點
#plt.plot(y_pred, y_pred, color= 'red', label= 'XGBoost model') #迴歸線
#plt.xlabel('Data')
#plt.ylabel('Quality')
#plt.title('XGB')
#plt.legend()
#plt.show()





#Lightgbm

#lgbrg=lgb.LGBMRegressor()
#lgbrg.fit(X_train,y_train)
#y_pred = lgbrg.predict(X_test) # Predictions
#y_true = y_test # True values
#MSE = mse(y_true, y_pred)
#RMSE = np.sqrt(MSE)
#R_squared = r2_score(y_true, y_pred)
#
#print(MSE)
#print(RMSE)
#print(R_squared)
#plt.scatter(y_pred, y_test, color= 'black', label= 'Data') #數據點
#plt.plot(y_pred, y_pred, color= 'red', label= 'lightgbm model') #迴歸線
#plt.xlabel('data')
#plt.ylabel('quality')
#plt.title('lightgbm')
#plt.legend()
#plt.show()
##RND
#from sklearn.ensemble import RandomForestRegressor
#rmdf=RandomForestRegressor()
#rmdf.fit(X_train,y_train)
#print("Regression",rmdf.score(X_test,y_test))
#y_pred = rmdf.predict(X_test) # Predictions
#y_true = y_test # True values
#MSE = mse(y_true, y_pred)
#RMSE = np.sqrt(MSE)
#R_squared = r2_score(y_true, y_pred)
#print(MSE)
#print(RMSE)
#print(R_squared)
#plt.scatter(y_pred, y_test, color= 'black', label= 'Data') #數據點
#plt.plot(y_pred, y_pred, color= 'red', label= 'RandomForest model') #迴歸線
#plt.xlabel('data')
#plt.ylabel('quality')
#plt.title('RandomForest')
#plt.legend()
#plt.show()



