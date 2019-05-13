# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:57:49 2019

@author: DimBottom
"""

import pandas as pd
import numpy as np

#采集数据
data_train = pd.read_csv(r"C:Users\87515\Desktop\train.csv")

#基本描述
#print(data_train.describe())
import os
if not (os.path.exists('DF')):
    os.mkdir('DF')
#data_train.describe().to_csv(r'DF\describe.csv')

#可视化分析
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize = (12,6))
fig.set(alpha = 0)
fig.set(dpi = 72)


##生存情况
#data_train.Survived.value_counts().plot(kind = 'bar', width = .2)
#plt.title('生存情况')
#S = data_train.Survived.value_counts()
#for index, y in enumerate(np.array(data_train.Survived.value_counts())):
#    plt.text(index, y+20, '%d' % y, ha='center', va= 'top', size = 14)
#plt.xticks(rotation = 0, size = 14)
#plt.savefig('DF\生存情况')
#plt.show()


##乘客等级
#S = data_train.Pclass
#S.value_counts().plot(kind = 'bar', width = .2)
#plt.title('乘客等级')
#for index, y in enumerate(np.array(S.value_counts())):
#    plt.text(index, y+20, '%d' % y, ha='center', va= 'top', size = 14)
#plt.xticks(rotation = 0, size = 14)
#plt.savefig('DF\乘客等级')
#plt.show()


##乘客性别
#S = data_train.Sex
#S.value_counts().plot(kind = 'bar', width = .2)
#plt.title('乘客性别')
#for index, y in enumerate(np.array(S.value_counts())):
#    plt.text(index, y+20, '%d' % y, ha='center', va= 'top', size = 14)
#plt.xticks(rotation = 0, size = 14)
##plt.savefig('DF\乘客性别')
#plt.show()


##Female&male与Survived关系
#S1 = data_train.Sex
#S2 = data_train.Survived
#
#plt.subplot(1,2,1)
#S1.value_counts().plot(kind='bar', color = 'g', width=.2, label='0')
#plt.xticks(rotation = 0, size = 14)
#plt.bar(range(0,2), S1[S2 == 1].value_counts(),color = 'b', width=.2,label='1')
#plt.legend()
#plt.title(r'Female&Male的Survived情况', size = 14)
#
#plt.subplot(1,2,2)
#S2.value_counts().plot(kind='bar', width=.2, color = 'g', label='Female')
#plt.xticks(rotation = 0, size = 14)
#plt.bar(range(0,2), S2[S1 == 'male'].value_counts(), color = 'b', width=.2, label='Male')
#plt.legend()
#plt.title('Survived的Female&Male情况')
#
#plt.savefig(r'DF\Female&male与Survived关系')


##年龄与生存的关系
#S1=data_train.Age
#S2=data_train.Surviveds
#s1=S1.value_counts().sort_index()
#s2=S1[S2 == 0].value_counts().sort_index()
#plt.bar(s1.index, s1, width = .6, label='获救')
#plt.bar(s2.index, s2, width = .6, label='未获救')
#plt.legend()
#plt.xticks(rotation = 0, size = 14)
#plt.xticks(range(0,85,5))
#plt.xlim(-1,81)
#plt.ylim(0,31)
#plt.title('年龄与生存的关系', size=14)
#plt.xlabel('年龄', size=14)
#plt.ylabel('')
#plt.savefig('DF\年龄与生存的关系')
#plt.show()


##不同等级船舱的年龄分布
#S1=data_train.Age
#S2=data_train.Pclass
#S1[S2==1].plot(kind='kde', label='头等舱')
#S1[S2==2].plot(kind='kde', label='二等舱')
#S1[S2==3].plot(kind='kde', label='三等舱')
#plt.xlabel('年龄',size=14)
#plt.ylabel('')
#plt.legend()
#plt.title('不同等级船舱的年龄分布', size=14)
#plt.savefig('DF\不同等级船舱的年龄分布')
#print('头等舱平均年龄:',S1[S2==1].mean())
#print('二等舱平均年龄:',S1[S2==2].mean())
#print('三等舱平均年龄:',S1[S2==3].mean())

##各乘客等级的获救情况
#S1=data_train.Pclass
#S2=data_train.Survived
#df=pd.DataFrame({u'获救':S1[S2 == 1].value_counts(), u'未获救':S1[S2 == 0].value_counts()})
#df.plot(kind='bar', stacked=True)
#plt.xticks(rotation=0)
#plt.title(u"各乘客等级的获救情况")
#plt.xlabel(u"乘客等级") 
#plt.ylabel(u"人数") 
#plt.savefig(r'DF\各乘客等级的获救情况')
#plt.show()


#S1 = data_train.Sex
#S2 = data_train.Age
#df = pd.DataFrame({u'女':S2[S1=='female'],u'男':S2[S1=='male']})
#print(S1[S1=='female'].count())
#print(S1[S1=='male'].count())
#print(df)
#df.plot(kind='bar')


##根据舱等级和性别的获救情况
#S1 = data_train.Sex
#S2 = data_train.Pclass
#S3 = data_train.Survived
#
#plt.subplot(141)
#plt.title(u"高级船舱女性的获救情况")
#plt.bar([0,1], S3[S1=='female'][S2!=3].value_counts().sort_index(), color='#FA2479', width=.5)
#plt.xticks([0,1],[u'未获救',u'获救'])
#plt.xlim([-.5,1.5])
#plt.yticks(range(0,350,100))
#plt.legend([u"女性/高级舱"], loc='best')
#
#plt.subplot(142)
#plt.title(u"低级船舱女性的获救情况")
#plt.bar([0,1], S3[S1=='female'][S2==3].value_counts().sort_index(), color='pink', width=.5)
#plt.xticks([0,1],[u'未获救',u'获救'])
#plt.xlim([-.5,1.5])
#plt.yticks(range(0,350,100))
#plt.legend([u"女性/低级舱"], loc='best')
#
#plt.subplot(143)
#plt.title(u"高级船舱男性的获救情况")
#plt.bar([0,1], S3[S1=='male'][S2!=3].value_counts().sort_index(), color='lightblue', width=.5)
#plt.xticks([0,1],[u'未获救',u'获救'])
#plt.xlim([-.5,1.5])
#plt.yticks(range(0,350,100))
#plt.legend([u"男性/高级舱"], loc='best')
#
#plt.subplot(144)
#plt.title(u"低级船舱男性的获救情况")
#plt.bar([0,1], S3[S1=='male'][S2==3].value_counts().sort_index(), color='steelblue', width=.5)
#plt.xticks([0,1],[u'未获救',u'获救'])
#plt.xlim([-.5,1.5])
#plt.yticks(range(0,350,100))
#plt.legend([u"男性/低级舱"], loc='best')
#
#plt.savefig(r'DF\根据舱等级和性别的获救情况')
#plt.show()

##各登录港口乘客的获救情况
#Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
#Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
#df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
#df.plot(kind='bar', stacked=True)
#plt.xticks(rotation=0)
#plt.title(u"各登录港口乘客的获救情况")
#plt.xlabel(u"登录港口") 
#plt.ylabel(u"人数") 
#plt.savefig(r"DF\各登录港口乘客的获救情况")
#plt.show()

##不同港口的船舱等级、性别情况
#S=data_train.Embarked
#S1=data_train.Pclass
#S2=data_train.Sex
#plt.subplot(131)
#plt.title(u'S港口')
#print(S1[S=='S'].value_counts().sort_index())
#plt.bar([0.5,0.6,0.7],S1[S=='S'].value_counts().sort_index(),width=0.1,color=['pink','lightgreen','lightblue'])
#plt.bar([1.5,1.6],S2[S=='S'].value_counts().sort_index(),width=0.1,color=['steelblue','#FA2479'])
#plt.xlim([0,2])
#plt.xticks([0.6,1.55],[u'船舱等级(1、2、3)',u'性别(女/男)'])
#plt.yticks(range(0,500,100))
#plt.ylabel(u'人数')
#
#plt.subplot(132)
#plt.title(u'C港口')
#plt.bar([0.5,0.6,0.7],S1[S=='C'].value_counts().sort_index(),width=0.1,color=['pink','lightgreen','lightblue'])
#plt.bar([1.5,1.6],S2[S=='C'].value_counts().sort_index(),width=0.1,color=['steelblue','#FA2479'])
#plt.xlim([0,2])
#plt.xticks([0.6,1.55],[u'船舱等级(1、2、3)',u'性别(女/男)'])
#plt.yticks(range(0,500,100))
#plt.ylabel(u'人数')
#
#plt.subplot(133)
#plt.title(u'Q港口')
#plt.bar([0.5,0.6,0.7],S1[S=='Q'].value_counts().sort_index(),width=0.1,color=['pink','lightgreen','lightblue'])
#plt.bar([1.5,1.6],S2[S=='Q'].value_counts().sort_index(),width=0.1,color=['steelblue','#FA2479'])
#plt.xlim([0,2])
#plt.xticks([0.6,1.55],[u'船舱等级(1、2、3)',u'性别(女/男)'])
#plt.yticks(range(0,500,100))
#plt.ylabel(u'人数')
#plt.savefig(r'DF\不同港口的船舱等级、性别情况')
#plt.show()

##不同港口的年龄分布
#S1=data_train.Age
#S2=data_train.Embarked
#S1[S2=='S'].plot(kind='kde', label='S')
#S1[S2=='C'].plot(kind='kde', label='C')
#S1[S2=='Q'].plot(kind='kde', label='Q')
#plt.xlabel('年龄',size=14)
#plt.ylabel('')
#plt.legend()
#plt.title('不同等级船舱的年龄分布', size=14)
#plt.savefig('DF\不同港口的年龄分布')

##船票费用和生存的关系
#S=data_train.sort_values('Fare')
#S1=S.Survived
#S2=S.Fare
#S1.loc[S1==0]=-1
#plt.scatter(S2,S1)
#plt.title('船票费用和生存的关系')
#plt.xlim([-5,515])
#plt.yticks([-1,1])
#ax=plt.gca()
#ax.spines['bottom'].set_position(('data',0))
#ax.spines['top'].set_color('none')
#plt.savefig(r'DF\船票费用和生存的关系')

##兄妹&配偶数和生存的关系
#S=data_train
#S1=S.Survived
#S2=S.SibSp[S1==1].value_counts().sort_index()
#S3=S.SibSp[S1==0].value_counts().sort_index()
##print(S2)
##print(S3)
#plt.bar(S2.index,S2,width=.4)
#plt.bar(S3.index,-S3,width=.4)
#plt.title('兄妹&配偶数和生存的关系')
#plt.xticks(range(0,9))
#ax=plt.gca()
#ax.spines['bottom'].set_position(('data',0))
#ax.spines['top'].set_color('none')
#plt.savefig(r'DF\兄妹&配偶数和生存的关系')

##子女&父母数和生存的关系
#S=data_train
#S1=S.Survived
#S2=S.Parch[S1==1].value_counts().sort_index()
#S3=S.Parch[S1==0].value_counts().sort_index()
##print(S2)
##print(S3)
#plt.bar(S2.index,S2,width=.4)
#plt.bar(S3.index,-S3,width=.4)
#plt.title('子女&父母数和生存的关系')
#plt.xticks(range(0,9))
#ax=plt.gca()
#ax.spines['bottom'].set_position(('data',0))
#ax.spines['top'].set_color('none')
#plt.savefig(r'DF\子女&父母数和生存的关系')

##按Cabin有无看获救情况
#Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
#Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
#df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
#df.plot(kind='bar', stacked=True)
#plt.title(u"按Cabin有无看获救情况")
#plt.xticks(rotation=0)
#plt.xlabel(u"Cabin有无") 
#plt.ylabel(u"人数")
#plt.savefig(r'DF\按Cabin有无看获救情况')
#plt.show()

##年龄和船票费用的关系
#S=data_train.sort_values('Age')
#S1=data_train.Age
#S2=data_train.Fare
#plt.scatter(S1,S2)
#plt.title('年龄和船票费用的关系')
#plt.xlabel(u'年龄')
#plt.ylabel(u'费用')
#plt.ylim(-1,550)
#plt.savefig(u'DF\年龄和船票费用的关系')
#plt.show()

##年龄和SibSp的关系
#S=data_train.sort_values('Age')
#S1=data_train.Age
#S2=data_train.SibSp
#plt.scatter(S1,S2)
#plt.title('年龄和SibSp的关系')
#plt.xlabel(u'年龄')
#plt.ylabel(u'SibSp')
#plt.savefig(u'DF\年龄和SibSp的关系')
#plt.show()

##年龄和Parch的关系
#S=data_train.sort_values('Age')
#S1=data_train.Age
#S2=data_train.Parch
#plt.scatter(S1,S2)
#plt.title('年龄和Parch的关系')
#plt.xlabel(u'年龄')
#plt.ylabel(u'Parch')
#plt.savefig(u'DF\年龄和Parch的关系')
#plt.show()

##年龄和性别的关系
#S=data_train.sort_values('Age')
#S1=S.Age
#S2=S.Sex
#S3=S1[S2=='female'].value_counts()
#S4=S1[S2=='male'].value_counts()
#plt.subplot(211)
#plt.title('年龄和性别的关系')
#plt.bar(S3.index, S3)
#plt.xlabel('女')
#plt.yticks(range(0,25,5))
#plt.ylabel('人数')
#plt.subplot(212)
#plt.bar(S4.index, S4, color='orange')
#plt.xlabel('男')
#plt.yticks(range(0,25,5))
#plt.ylabel('人数')
#plt.savefig(u'DF\年龄和性别的关系')
#plt.show()








