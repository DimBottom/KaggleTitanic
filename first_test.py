# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:50:11 2019

@author: 87515
"""

import pandas as pd
import numpy as np

data_train=pd.read_csv(r"C:\Users\87515\Desktop\train.csv")
#Passengerld, Survived, Pclass, Name, Sex, Age, SibSp
#Parch, Ticket, Fare, Cabin, Embarked


#用于分类回归
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
#    print(known_age)
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # y即目标年龄
    y = known_age[:, 0]
#    print(y)
    
    # X即特征属性值
    X = known_age[:, 1:]
#    print(X)
    
    #random_state表示一个种子的值，n_estimators表示子树的值，n_jobs表示最大的处理器，-1表示不限制
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # fit到RandomForestRegressor之中
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr
    

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

#缺失值处理
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

#数据预处理

#数据因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

#合成数据集
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#删除指定数据集，axis为0(index) or 1(columns),inpalace表示是否对内部操作，如果True则不返回
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

#处理数据集中浮动较大的数值到(-1,1)之间
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
#fit的数据需要以2维([[]])的形式传入
age_scale_param = scaler.fit(df[['Age']])
#将fit的数据处理后的结果以np.array的形式返回
df['Age_scaled'] =age_scale_param.transform(df[['Age']])
fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = fare_scale_param.transform(df[['Fare']])
df['Age']=df['Age_scaled']
df['Fare']=df['Fare_scaled']
#df.to_csv('look.csv')
#inddex(None) Passenger Age SibSp Parch Fare Cabin_No Cabin_Yes
#Embarked_C Embaked_Q Embarked_S Sex_female Sex_male
#Pclass_1 Pclass_2 Pclass_3 Age_scaled Fare_scaled

#筛选数据(结果+特征)
#利用正则表达式
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#train_df.to_csv('look_filter.csv')
train_np = train_df.as_matrix()


#数据建模

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

from sklearn import linear_model
# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

#测试集预处理

data_test = pd.read_csv(r'C:\Users\87515\Desktop\test.csv')

#缺失值处理
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)

#向量化
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

#合成新的数据集
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

#处理浮动值大的数据
age_scale_param = scaler.fit(df_test[['Age']])
df_test['Age_scaled'] =age_scale_param.transform(df_test[['Age']])
fare_scale_param = scaler.fit(df_test[['Fare']])
df_test['Fare_scaled'] = fare_scale_param.transform(df_test[['Fare']])
df_test['Age']=df_test['Age_scaled']
df_test['Fare']=df_test['Fare_scaled']

#筛选数据(特征)
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test = test.as_matrix()

#结果预测
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("first_test_reslut.csv", index=False)




















