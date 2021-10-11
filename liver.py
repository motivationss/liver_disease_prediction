import pandas as pd
import matplotlib.pyplot as plt
cm = plt.cm.get_cmap('RdYlBu_r')
import seaborn as sns



data = pd.read_excel(r'C:\Users\86188\Desktop\young\cs\gt\Indian Liver Patient Dataset (ILPD).xlsx')
c = data['gender']
c[c =='Male' ] = 0
c[c =='Female'] = 1 
data['gender'] = c
data.dropna(inplace = True)
#data = data.dropna(inplace = False)

a = data.describe() 
data.hist(figsize=(12,8))


##################
#calculate correlation

dfData = data.corr().round(2)

plt.subplots(figsize=(7, 7)) # set image size

#cm = plt.cm.get_cmap('RdYlBu')
m = sns.heatmap(dfData, annot=True, linewidths = 0.05,vmax=1, square=True, cmap=cm)
#cbar = plt.colorbar(m)
#cbar.set_label('$T_B(K)$',fontdict=font)
#cbar=m.colorbar(im,'bottom',size=0.2,pad=0.3,label='W*m-2')

#cbar.set_ticks(np.linspace(-1,1,0.1))
plt.rcParams['font.sans-serif']=['SimHei'] #used to indicate labels
plt.rcParams['axes.unicode_minus']=True #used to indicate minus signs
#plt.savefig('./BluesStateRelation.png')
plt.show()



###########

#Prediction
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np



#data = data.loc[0:10,:].copy()
y = data.loc[:,12]
data.drop(12,axis=1, inplace=True)
x = data

# 
#print('##################################################################')
# 
# SVM
X_train,X_test,y_train, y_test= train_test_split(x, y,train_size=0.8, random_state=5)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, classification_report,roc_auc_score

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
model6 = GridSearchCV(svc, parameters, cv=2)

#clf =svm.SVC(kernel='rbf')
model6.fit(X_train, y_train)
pred6 = model6.predict(X_test)
svcaccuracy = accuracy_score(y_test,pred6)
svcprecision = precision_score(y_test,pred6)
svcrecall = recall_score(y_test,pred6)
svcroc=roc_auc_score(y_test,pred6)
svcf1_score = f1_score(y_test,pred6 )




##################

##decisontree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#score = clf.score(X_test,y_test.ravel())
#print('the score is :', score)

from sklearn.ensemble import RandomForestClassifier
#clf.fit(X_train, y_train)
#score = clf.score(X_test,y_test.ravel())
#print('the score is :', score)


param_grid =[ {"max_depth": range(10,50,15),#don't consider when there's small amount of data/labels, normally between 10-100
              #"min_samples_split": [5, 10,15,20,25],#by default 2, used in large sample size
              #"min_samples_leaf": [5, 10,15,20,25],#by default 1
              #"bootstrap": [True, False],
              #"criterion": ["gini", "entropy"],
              "n_estimators": range(10,50,15),
              }
              # "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
              # "max_features": if there are few features (e.g <50), could use "None" by default; otherwise log, sqrt, auto
              # "warm_start": [True, False],
              # "oob_score": [True, False],
              # "verbose": [True, False]}
            ]
clf = RandomForestClassifier()

model7 = GridSearchCV(clf, param_grid, cv=3)

#clf =svm.SVC(kernel='rbf')
model7.fit(X_train, y_train)
pred7 = model7.predict(X_test)
rfaccuracy = accuracy_score(y_test,pred7)
rfprecision = precision_score(y_test,pred7)
rfrecall = recall_score(y_test,pred7)
rfroc=roc_auc_score(y_test,pred7)
rff1_score = f1_score(y_test,pred7 )


model_performance = pd.DataFrame({
    "model": ['SVM','Random Forest'],
    "accuracy": [
              svcaccuracy,  rfaccuracy, ],
    "precision": [
              svcprecision, rfprecision],
    "recall": [
              svcrecall,  rfrecall],  
    "f1_score": [
              svcf1_score, rff1_score],
    "roc": [
              svcroc,  rfroc]
})
