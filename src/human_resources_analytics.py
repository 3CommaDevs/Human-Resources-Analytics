#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:38:57 2017

@author: ricky_xu
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Modeling import Classifier
# from mlens.visualization import pca_plot,pca_comp_plot

#plot sales和slary的情况
def plot_categorical_features(plot=False):
    if plot == True:
        fig,axes = plt.subplots(ncols=2,figsize=(12,6))
        ax = sns.countplot(data['sales'],ax = axes[0])
        plt.setp(ax.get_xticklabels(),rotation=45)
        ax = sns.countplot(data['salary'], ax = axes[1])
        plt.tight_layout()
        plt.show()

#plot binary features(0,1): workAccident promotion left
def plot_binary_features(plot=False):
    if plot == True:
        fig,axes = plt.subplots(ncols=3,figsize=(12,6))
        # sns.countplot(data['workAccident'], ax = axes[0])
        ax = sns.countplot(x='workAccident', hue='left', data= data, ax=axes[0])
        ax.legend(labels=['stay','left'])
        ax = sns.countplot(x='promotion', hue = 'left',data = data, ax = axes[1])
        ax.legend(labels=['stay','left'])
        ax = sns.countplot(data['left'], ax = axes[2])
        ax.legend(labels=['stay','left'])
        plt.tight_layout()
        plt.show()

#plot numerical features
def plot_numerical_features(plot=False):
    if plot == True:
        fig,axes = plt.subplots(ncols=3,figsize=(12,6))
        sns.distplot(data['satisfaction'], ax = axes[0])
        sns.distplot(data['evaluation'], ax = axes[1])
        sns.distplot(data['averageMonthlyHours'], ax = axes[2])
        plt.tight_layout()
        plt.show()

#plot other features
def plot_other_features(plot=False):
    if plot == True:
        fig,axes = plt.subplots(ncols=2,figsize=(12,6))
        ax = sns.countplot(data['projectCount'],ax = axes[0])
        ax = sns.countplot(data['yearsAtCompany'], ax = axes[1])
        plt.tight_layout()
        plt.show()


def plot_feature_corr(plot=False):
    if plot == True:
        ax = sns.heatmap(data.corr(),annot=True,cmap="YlGnBu")
        plt.setp(ax.get_xticklabels(),rotation=10)
        plt.setp(ax.get_yticklabels(),rotation=10)
        plt.show()

left_labels={0:'Stay',1:'Left'}

def plot_left_feature(feature,plot=False):
    if plot==True:
        fig, axes = plt.subplots(ncols=3, figsize=(17, 6))
        data['left'] = data['left'].replace(left_labels)
        data_stay = data[data['left'] == 'Stay']
        data_left = data[data['left'] == 'Left']
        sns.boxplot(x='left', y=feature, data=data, ax=axes[0])
        axes[1].hist(data_stay[feature], bins=6, label='Stay', alpha=0.7)
        axes[1].hist(data_left[feature], bins=6, label='Left', alpha=0.7)
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Count')
        axes[1].legend()
        ax = sns.kdeplot(data=data_stay[feature], color='b', shade=True, ax=axes[2])
        ax = sns.kdeplot(data=data_left[feature], color='r', shade=True, ax=axes[2])
        ax.legend(['Stay', 'Left'])
        ax.set_xlabel(feature)
        ax.set_ylabel('Dentisty')
        plt.tight_layout()
        plt.show()

#plot left and satisfaction
def plot_left_satisfaction(feature='satisfaction',plot=False):
    plot_left_feature(feature, plot)



def cut_satisfaction():
    data['satisfaction_leval']=np.where(data['satisfaction']<0.2,0,
                                                 np.where(data['satisfaction']>=0.5,2,1))
    return data

#plot left and projectCount
def plot_left_projectCount(feature='projectCount',plot=False):
    plot_left_feature(feature, plot)

def cut_projectCount():
    data['projectCount_leval']=np.where(data['projectCount']<=2,0,
                                                 np.where(data['projectCount']>5,2,1))
    return data


#plot left and evaluation
def plot_left_evaluation(feature='evaluation',plot=False):
    plot_left_feature(feature, plot)


def cut_evaluation():
    def evaluation_level(i):
        if i<0.4:
            return 0
        elif i>=0.4 and i<0.576:
            return 1
        elif i>=0.576 and i<0.82:
            return 2
        else:
            return 3
    data['evaluation_leval']=data['evaluation'].apply(evaluation_level)
    return data


#plot left and averageMonthlyHours
def plot_left_averageMonthlyHours(feature='averageMonthlyHours',plot=False):
    plot_left_feature(feature, plot)


def cut_averageMonthlyHours():
    def evaluation_level(i):
        if i<111:
            return 0
        elif i>=111 and i<162:
            return 1
        elif i>=162 and i<242:
            return 2
        else:
            return 3
    data['averageMonthlyHours_leval']=data['averageMonthlyHours'].apply(evaluation_level)
    return data

#plot left and yearsAtCompany
def plot_left_yearsAtCompany(feature='yearsAtCompany',plot=False):
    plot_left_feature(feature, plot)

def cut_yearsAtCompany():
    data['yearsAtCompany_leval']=np.where(data['yearsAtCompany']<=3, 0,
                                          np.where(data['yearsAtCompany']<=6,1,2))
    return data

def plot_avg_hour_project(feature='avg_hour_project',plot=False):
    plot_left_feature(feature, plot)

def cut_avg_hour_project():
    data['avg_hour_project_leval']=np.where(data['avg_hour_project']<495,0,
                                            np.where(data['avg_hour_project']<608,1,
                                                     np.where(data['avg_hour_project']<742,2,
                                                              np.where(data['avg_hour_project']<988,3,4))))
    return data


def plot_evaluation_project(feature = 'evaluation_project', plot =False):
    plot_left_feature(feature, plot)

def cut_evaluation_project():
    data['evaluation_project_leval'] = np.where(data['evaluation_project']<0.128,0,
                                                np.where(data['evaluation_project']<0.1488,1,
                                                         np.where(data['evaluation_project']<0.2215,2,
                                                                  np.where(data['evaluation_project']<0.294,3,4))))
    return data

def plot_evaluation_avg_hour(feature = 'evaluation_avg_hour',plot=False):
    plot_left_feature(feature, plot)

def cut_evaluation_avg_hour():
    data['evaluation_avg_hour_leval']=np.where(data['evaluation_avg_hour']<0.086,0,
                                               np.where(data['evaluation_avg_hour']<0.127,1,2))
    return data

def plot_sales(feature='sales',plot=False):
    plot_left_feature(feature,plot)

def cut_data():
    #based on satisfaction_dentisty  cut satisfaction
    data = cut_satisfaction()

    #based on projectCount_dentisty  cut projectCount
    data = cut_projectCount()

    #based on evalution_dentisty  cut evalution
    data = cut_evaluation()
    
    #based on averageMonthlyHours_dentisty  cut averageMonthlyHours
    data = cut_averageMonthlyHours()
    data = cut_yearsAtCompany()
    
    #from plot cut leval of avg_hour_project
    data = cut_avg_hour_project()
    data = cut_evaluation_project()
    data = cut_evaluation_avg_hour()
    
    return data 

def cut_sales():
    data['sales_leval'] = np.where(data['sales']<=1,0,
                                   np.where(data['sales']<=3,1,
                                            np.where(data['sales']<=5,2,3)))
    
    del data['sales']
    return data
    
def numerical_string_feature(columns):
    if type(columns) !=list:
        columns = list(columns)
    for col in columns:
        col_feature= list(data[col].unique())
        data[col] = data[col].map(dict(zip(col_feature,range(len(col_feature)))))
    return data

def scale_data(columns,scale='Standard'):
    
    scaler = StandardScaler() if scale=='Standard' else MinMaxScaler()
    
    for column in columns:
            data[column] = scaler.fit_transform(data[column])
    return data


def dummies(data, columns):
    for column in columns:
        data[column] = data[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in data[column].unique()]
        data = pd.concat((data, pd.get_dummies(data[column], prefix = column)[good_cols]), axis = 1)
        del data[column]
    return data

#获取X（features）Y(Target)
def get_x_y(data):
    X=data.drop(['left'],axis=1)
    y=data['left']
    return X,y

def before_modeling():
    pca_comp_plot(X_train,y=y_train,cmap='Set2')

def plot_feature_importance_final_model(plot = False):
    xgbc.train()
    xgbc.plot_feature_importance(plot)

def plot_learning_curve_final_model(plot=False):
    xgbc.train()
    xgbc.plot_learning_curve(cv=5,plt=plot)

if __name__ == "__main__":

#1.load data
    data=pd.read_csv("../data/HR_comma_sep.csv")
#    print (data)

#2. 查看data信息没有缺失值 10个features中 8个numerical features 2个categorical features
#    print (data.info())

#3. Renaming certain columns for better readability
    data = data.rename(columns={'satisfaction_level': 'satisfaction',
                                'last_evaluation': 'evaluation',
                                'number_project': 'projectCount',
                                'average_montly_hours': 'averageMonthlyHours',
                                'time_spend_company': 'yearsAtCompany',
                                'Work_accident': 'workAccident',
                                'promotion_last_5years': 'promotion',
                        })

#4. visualize each feature 

#    plot_categorical_features(plot=True)
#    plot_binary_features(plot=True)
#    plot_numerical_features(plot=True)
#    plot_other_features(plot=True)
#    plot_feature_corr(True)




#5. visualize the relationship between each feature and left situation

#a. satisfaction 
#    plot_left_satisfaction(plot=True)

#b. projectCount
#    plot_left_projectCount(plot=True)

#c. evalution
#    plot_left_evaluation(plot=True)

#d. averageMonthlyHours
#    plot_left_averageMonthlyHours(plot=True)

#e. yearsAtCompany
#    plot_left_yearsAtCompany(plot=True)

#6. create new feature 
#a. create new feature avg_hour_project
    data['avg_hour_project'] = (data['averageMonthlyHours'] * 12) / data['projectCount']

#b. create new feature evaluation_project
    data['evaluation_project'] = data['evaluation']/data['projectCount']
    
#c. creat new feature evaluation_avg_hour
    data['evaluation_avg_hour'] = data['evaluation']/(data['averageMonthlyHours']/30)
    
#7. visualize the new feature with left situation
#a. avg_hour_project
#    plot_avg_hour_project(plot=True)

#b. evaluation_project
#    plot_evaluation_project(plot=True)

#c. evaluation_avg_hour
#    plot_evaluation_avg_hour(plot=True)


#8. transform continuous variable into categorical variable
    data = cut_data()

#9. transform sting variable into numberical variable ( sales salary)
    columns=['sales','salary']
    data = numerical_string_feature(columns)
    data = cut_sales()
    
#10. normalize the value of some specific variable (make the value range from 0 to 1)
    columns =['avg_hour_project','averageMonthlyHours']
    data =scale_data(columns,scale='Maxmin')
    

#11. extract features and target of dataset (X_train means features; y_train means target)
    X_train,y_train=get_x_y(data)

#12. visualize the distribution of dataset 
#find the data is overlopped  can't use linear model --->use tree model
#    before_modeling()


##13. modeling
    # the parameter candidates used for GridSearch
    xgb_parameters={'colsample_bytree':[0.2,0.3,0.4],
                    'learning_rate':[0.05,0.1],
                    'max_depth':[4,8],
                    'n_estimators':[100,200,300],
                    'seed': [1337,1234]}
    
    #the final parameter of the model (XGBClassifier)
    xgb_params = {'learning_rate':0.1,
                  'max_depth': 9, 
                  'min_child_weight': 1,
                  'n_estimators':1000,
                  'gamma':0.4,
                  'subsample':0.95,
                  'colsample_bytree':0.8,
                  'scale_pos_weight':1,
                  'seed':1337}

#a. create a Classifier object 
    xgbc =Classifier(xgb.XGBClassifier,'xgbc',X_train,y_train,seed=0,params=xgb_params,scoring='accuracy')
    print ("train")
    xgbc.train()
#b. use GridSearch for finding best parameter (But it's not necessary since we have already find the best parameter)
#    xgbc.GridSearch(xgb_parameters)

#c. get the CrossValidationScore(the assesment criterion is 'f1')
    print ("Over")
    print (xgbc.CrossValScore(mean=True))

#d. visualize the feature_importance 
#    plot_feature_importance_final_model(plot = True)

#e. visualize the learning_curve to diagnose the model (whether the model is underfitting or overfitting)
#    plot_learning_curve_final_model(plot=True)
    



    
    



    


