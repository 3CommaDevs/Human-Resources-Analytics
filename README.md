Human Resources Analytics
=========================
![](images/logo.jpg)

Desciption
----------
The Human Resources Analytics is a dataset providing informations on the situation and work of several ten of thousands employees. In this task i'll focus on one very important question : Why employees are leaving the company ? To tackle this question , this solution combines data exploration analysis and modeling. This dataset is perfect for this kind of detailed data exploration because it contains a few number of features a large number of individual, so we can perform robust statistics. Firstlty, ill globally explore the dataset, then ill focus on a detailed exploration analysis of the stayed/left employees and i'll end by the data modeling. Finally, this solution gets the accuracy 
***99.0666%*** .  More details are in [***report***](Report.md).

Dependencies
-----------
* Numpy
* Pandas 
* scikit-learn
* matplotlib
* seaborn


Document Describation
---------------------

### data
- HR_comma_sep.csv :  training data.


### src
- human_resources_analytics.py : illustrate the process of handling and visulizing features.  And use machine learning method to do the classification.
- Modeling.py : encapsulate some commone functions in sklearn in order to train a model; 

Getting Started
---------------

Expalain how to run the function of the model

### Reminder

```
Please run the functions in the main function (if __name__ == "__main__");
And the function can work after dropping '#'
If you just want to see the performace of the model, you can direct to 13.Modeling
```

#### 1.Load the data

```
data=pd.read_csv("HR_comma_sep.csv")
```

#### 2.Breif inforamtion of data

```
print (data.info())
```

#### 3.Renaming certain columns for better readability

```
data = data.rename(columns={'satisfaction_level': 'satisfaction',
                                'last_evaluation': 'evaluation',
                                'number_project': 'projectCount',
                                'average_montly_hours': 'averageMonthlyHours',
                                'time_spend_company': 'yearsAtCompany',
                                'Work_accident': 'workAccident',
                                'promotion_last_5years': 'promotion',
                        })
```

#### 4\. Visualize Each Feature

```
plot_categorical_features(plot=True)
plot_binary_features(plot=True)
plot_numerical_features(plot=True)
plot_other_features(plot=True)
plot_feature_corr(True)
```

#### 5\. Visualize The Relationship Between Each Feature And Left Situation

```
#a. satisfaction 
plot_left_satisfaction(plot=True)

#b. projectCount
plot_left_projectCount(plot=True)

#c. evalution
plot_left_evaluation(plot=True)

#d. averageMonthlyHours
plot_left_averageMonthlyHours(plot=True)

#e. yearsAtCompany
plot_left_yearsAtCompany(plot=True)
```

#### 6\. Create New Feature

```
#a. create new feature avg_hour_project
data['avg_hour_project'] = (data['averageMonthlyHours'] * 12) / data['projectCount']

#b. create new feature evaluation_project
data['evaluation_project'] = data['evaluation']/data['projectCount']

#c. creat new feature evaluation_avg_hour
data['evaluation_avg_hour'] = data['evaluation']/(data['averageMonthlyHours']/30)
```

#### 7.Visualize The New Feature With Left Situation

```
#a. avg_hour_project
plot_avg_hour_project(plot=True)

#b. evaluation_project
lot_evaluation_project(plot=True)

#c. evaluation_avg_hour
plot_evaluation_avg_hour(plot=True)
```

#### 8\. Transform continuous variable into categorical variable

```
data = cut_data()
```

#### 9\. Transform string variable into numberical variable ( sales salary)

```
columns=['sales','salary']
data = numerical_string_feature(columns)
data = cut_sales()
```

#### 10\. Normalize the value of some specific variable (make the value range from 0 to 1)

```
columns =['avg_hour_project','averageMonthlyHours']
data =scale_data(columns,scale='Maxmin')
```

#### 11\. Extract features and target of dataset (X_train means features; y_train means target)

```
X_train,y_train=get_x_y(data)
```

#### 12\. Visualize the distribution of dataset

Find the data is overlopped can't use linear model --->use tree model

```
before_modeling()
```

#### 13\. Modeling

```
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
```

`A. Create a Classifier object`

```
xgbc =Classifier(xgb.XGBClassifier,'xgbc',X_train,y_train,seed=0,params=xgb_params,scoring='f1')
```

`B. Use GridSearch for finding best parameter (But it's not necessary since we have already find the best parameter)`

```
xgbc.GridSearch(xgb_parameters)
```

`C. Get the CrossValidationScore(the assesment criterion is 'f1')`

```
print (xgbc.CrossValScore(mean=True))
```

`D. Visualize the Feature_importance`

```
plot_feature_importance_final_model(plot = True)
```

`E.Visualize the Learning_Curve to diagnose the model (whether the model is underfitting or overfitting)`

```
plot_learning_curve_final_model(plot=True)
```

