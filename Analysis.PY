from pyexpat import model
from re import S
from cv2 import flip
from prometheus_client import Counter
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.metrics import r2_score
from sqlalchemy import column
from sympy import im
from xgboost import XGBClassifier, XGBRegressor
import imp
import lightgbm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report
import seaborn as sns
from sklearn.decomposition import PCA
#
#dataset=pd.read_csv(r"F:/DL/FCS.data_clean_v2.csv")
#df=pd.DataFrame(dataset)
##features=df.drop(["R"],axis=1) #特徵
##targets=df["R"] #目標
#pca_target=pd.DataFrame(df,columns=['B','D','F','I','Q'])
##print(pca_target)
#pca=PCA(n_components=1)
#pca_final=pca.fit_transform(pca_target) #降至一維
#minimax=preprocessing.MinMaxScaler()
#data_minimax=minimax.fit_transform(pca_final)
##pca_transform=pca.inverse_transform(pca_final) #轉為原始數據
##print(type(pca_final),pca_final.shape)
#pca_data=data_minimax
#df2=pd.DataFrame(pca_data,columns=['S'],dtype=float)
##print(df2)
#pca_final_data=pd.merge(df,df2,left_index=True,right_index=True)
##print(pca_final_data)
#new_data=pd.DataFrame(pca_final_data,columns=['A','C','E','G','H','J','K','L','M','N','O','P','S','R'])
#new_data_fillna=new_data.fillna(0)
##print(new_data)
#features=new_data_fillna.drop(["R"],axis=1) #特徵
#targets=new_data_fillna["R"] #目標
#
#X_train,X_test,y_train,y_test=train_test_split(features,targets,train_size=0.80) #資料分割
##smote
##X_res,y_res=SMOTE(random_state=42).fit_resample(X_train,y_train)
#
##XGBOOST
#xgrg=XGBClassifier().fit(X_train,y_train)
#y_pred=xgrg.predict(X_test)
#print("ROC_report",roc_auc_score(y_test,y_pred))
#print("ACC",accuracy_score(y_pred,y_test))
##視覺化
#corrmat = new_data_fillna.corr()
#f, ax = plt.subplots(figsize=(18, 18))
#sns.heatmap(corrmat, vmax=.8, square=True,annot=True)
#plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:/DL/FCS.data_clean_Lin.csv")
df=pd.DataFrame(dataset)
features=df.drop(["N"],axis=1) #特徵
targets=df["N"] #目標


#pca_target=pd.DataFrame(df,columns=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q'])
#print(pca_target)
#pca=PCA(n_components=10)
#pca_final=pca.fit_transform(pca_target) #降至一維
#print(pca_final)
##pca_transform=pca.inverse_transform(pca_final) #轉為原始數據
##print(type(pca_final),pca_final.shape)
#pca_data=pca_target.fillna(0)
#df2=pd.DataFrame(pca_data,columns=['S'],dtype=float)
##print(df2)
#pca_final_data=pd.merge(df,df2,left_index=True,right_index=True)
#print(pca_final_data)


#new_data=pd.DataFrame(pca_final_data,columns=['S','R'])
#new_data_fillna=new_data.fillna(0)
#features=new_data_fillna.drop(["R"],axis=1) #特徵
#targets=new_data_fillna["R"] #目標
#
X_train,X_test,y_train,y_test=train_test_split(features,targets,train_size=0.70) #資料分割
##smote
##X_res,y_res=SMOTE(random_state=42).fit_resample(X_train,y_train)
#
##XGBOOST
mode1=XGBClassifier().fit(X_train,y_train)
y_pred_xgrg=mode1.predict(X_test)
print("XGB_ROC_report",roc_auc_score(y_test,y_pred_xgrg))
print("XGB_ACC",accuracy_score(y_pred_xgrg,y_test))
print("XGB_f1",f1_score(y_pred_xgrg,y_test))
#RNDF
model2=RandomForestClassifier().fit(X_train,y_train)
y_pred_RNDF=model2.predict(X_test)
print("RNDF_ROC_report",roc_auc_score(y_test,y_pred_RNDF))
print("RNDF_ACC",accuracy_score(y_pred_RNDF,y_test))
print("RNDF_f1",f1_score(y_pred_RNDF,y_test))
#lightgbm
model3=lightgbm.LGBMClassifier().fit(X_train,y_train)
y_pred_LGBM=model3.predict(X_test)
print("LGBM_ROC_report",roc_auc_score(y_test,y_pred_LGBM))
print("LGBM_ACC",accuracy_score(y_pred_LGBM,y_test))
print("LGBM_f1",f1_score(y_pred_LGBM,y_test))
##視覺化
#corrmat = new_data_fillna.corr()
#f, ax = plt.subplots(figsize=(18, 18))
#sns.heatmap(corrmat, vmax=.8, square=True,annot=True)
#plt.show()

rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =0, min_samples_leaf = 10)
rfc.fit(X_train,y_train)
#print(rfc.score(X_test,y_test))

importance=rfc.feature_importances_
indices = np.argsort(importance)[::-1]
features = X_train.columns


from xgboost import XGBClassifier
from sklearn import svm
from sklearn.svm import SVC
import pandas as pd

#隨機森林
importances = rfc.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
#plt.show()

feature_score=pd.Series(importances,index=features).sort_values(ascending=False)
print(feature_score)
print(X_train)
from sklearn.metrics import confusion_matrix , roc_curve
#cnf_matrix = confusion_matrix(y_test, y_pred)
#XGRG
confmat=confusion_matrix(y_test, y_pred_xgrg, labels=[0,1])
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.show()
#RNDF
confmat=confusion_matrix(y_test, y_pred_RNDF, labels=[0,1])
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.show()
#LGBM
confmat=confusion_matrix(y_test, y_pred_LGBM, labels=[0,1])
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.show()


import scikitplot as skplt
prob = mode1.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(y_test, prob)
plt.show()
prob = model2.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(y_test, prob)
plt.show()
prob = model3.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(y_test, prob)
plt.show()