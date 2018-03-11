#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:38:58 2017

@author: ricky_xu
"""

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import VotingClassifier,BaggingRegressor
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score,GridSearchCV,learning_curve


def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,scoring='f1',n_jobs=1,
                        train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,scoring=scoring,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
    
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('training sizes')
        plt.ylabel('score')
#        plt.gca().invert_yaxis()
        plt.grid()
        
        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,
                         alpha=0.1,color='b')
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,
                         alpha=0.1,color='r')
        plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='score in training set')
        plt.plot(train_sizes,test_scores_mean,'o-',color='r',label='score in validation set')
        
        plt.legend(loc='best')
        
    plt.show()
    
class Classifier(object):
    def __init__(self,clf,name,X_train,y_train,seed=0,params=None,scoring='f1'):
#        params['random_state']=seed
        self.name=name
        self.clf=clf(**params)
        self.X=X_train
        self.y=y_train
        self.scoring =  scoring 
        
    def train(self):
        self.clf.fit(self.X,self.y)
    
    def predict(self,x):
        return self.clf.predict(x)
    
    def GridSearch(self,params,cv=5):
        clf=GridSearchCV(self.clf,params,cv=cv,scoring=self.scoring,verbose=1)
        clf.fit(self.X,self.y)
        best_parms=clf.best_params_
        best_score=clf.best_score_
        best_model=clf.best_estimator_
        self.clf=best_model
        print (best_score)
        print (best_parms)
    
    def plot_learning_curve(self,cv,plt=False):
        
        if plt==True:
            plot_learning_curve(self.clf,cv=cv,scoring=self.scoring,title='learning curve',X=self.X,y=self.y)
    
    def CrossValScore(self,mean=False):
        if mean==True:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring=self.scoring).mean()
        else:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring=self.scoring)
    def feature_importance(self,max_features=40):
        if self.clf.feature_importances_.any():
            indices=np.argsort(self.clf.feature_importances_)[::-1][:max_features]
            feature_importances=self.clf.feature_importances_[indices][:40]
            Features=self.X.columns[indices][:max_features]
            pd_feature_importances=pd.DataFrame({'Features':Features,'Feature_Importances':feature_importances})
        else:
            print ("valid feature_importance")
            pd_feature_importances=None
        return pd_feature_importances
    
    def plot_feature_importance(self,plot=False):
        if plot == True:
            ax = sns.barplot(x='Features', y='Feature_Importances', data=self.feature_importance())
            plt.setp(ax.get_xticklabels(), rotation=40)
            plt.show()

            

class Ensemble():
    def __init__(self,classifiers,X_train,y_train):
        self.classifiers=classifiers
        self.clfs=[clf.clf for clf in self.classifiers]
        self.clf_names=[clf.name for clf in self.classifiers]
        self.X=X_train
        self.y=y_train
        self.clf=None
#    def simple_modeling(self,plt=False):
#        cv_results=[cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=10) for clf in classifiers]
#        cv_means=[result.mean() for result in cv_results]
#        cv_std=[result.std() for result in cv_results]
#        
#        cv_res=pd.DataFrame({'CrossValMeans':cv_means,'CrossValerrors':cv_std,'Algorithm':['SVC','DecisonTree','AdaBoost','RandomForest','ExtraTrees','GradientBoost','LogisticRegression']})
#    
#        g=sns.barplot('CrossValMeans','Algorithm',data=cv_res,palette='Set3',orient='h',**{'xerr':cv_std})
#        g.set_xlabel('Mean_Accuracy')
#        g=g.set_title('Cross_validation_scores')
#    
#        plt.show()
    
    def votingC(self):
        votingC=VotingClassifier(estimators=[(name,clf) for name,clf in zip(self.clf_names,self.clfs)],
                                         voting='soft')
        votingC.fit(self.X,self.y)
        self.clf=votingC
        return votingC
    
#    def bagging_clf(self):
        
    
    def plot_final_learning_curve(self,cv,plt=True):
        if plt==True:
            plot_learning_curve(self.clf,cv=cv,title='learning curve',X=self.X,y=self.y) 
    
    def CrossValScore(self,mean=False):
        if mean==True:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring='accuracy').mean()
        else:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring='accuracy')