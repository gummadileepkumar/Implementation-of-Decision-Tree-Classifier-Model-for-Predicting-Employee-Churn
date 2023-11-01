# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
### Developed by: Gumma Dileep Kumar
### RegisterNumber:  212222240032
```python

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### Data head():
![ml_5 1](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/365dfa76-a06e-493a-9433-ef9fb042e31d)



### Data set info():

![ml_5 2](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/99a17201-170f-4148-acb8-762e420a2133)


### Null dataset:

![ml_5 3](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/58adcb54-d50c-4cbf-8e0a-a0fe7e3f92d0)


### Values count():

![ml_5 4](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/274933cc-f9d4-467a-b941-262d4fd5edb1)



### Data head() for salary:

![ml_5 5](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/6708f606-36b8-463a-be61-847f49205261)


### x.head():

![ml_5 6](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/46f1c855-cd6b-431d-b4f8-2344c6bfdbc9)


### Accuracy value:

![ml_5 7](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/affbddc1-113b-47b7-8cdf-6c8c5dc55743)


### Data prediction:
![ml_5 8](https://github.com/gummadileepkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707761/615d7c6b-5b67-4b66-aed4-ffd7c48fff09)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
