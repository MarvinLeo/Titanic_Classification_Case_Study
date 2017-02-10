import pandas as pd
import numpy as np
from sklearn.preprocessing import scale



##read data, drop missing and unrelative data
train_df = pd.read_csv("train.csv", index_col='PassengerId')
test_df = pd.read_csv("test.csv", index_col="PassengerId")
test_label = pd.read_csv("gender_submission.csv", index_col="PassengerId")
train_df['Train'] = 1
test_df['Train'] = 0
test_df['Survived'] = test_label['Survived']
train_df = train_df.append(test_df)
colname = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Train']
train_df.drop('Name', axis=1, inplace=True)
train_df.drop('Ticket', axis=1, inplace=True)
train_df.drop('Cabin', axis=1, inplace=True)
# print train_df.head()
#print train_df.columns

for i in enumerate(colname):
    print i, len(set(train_df[colname[i[0]]]))

#replace data with number
map = {'male': 0, 'female':1}
train_df['Sex'].replace(map, inplace=True)
map = {'Q':0, 'S':1, 'C':2}
train_df['Embarked'].replace(map, inplace=True)
train_df = train_df[~pd.isnull(train_df['Age'])]
train_df = train_df[~pd.isnull(train_df['Embarked'])]
train_df = train_df[~pd.isnull(train_df['Fare'])]

## grap X and y
train_label = train_df[train_df['Train'] == 1]['Survived'].values.tolist()
test_label = train_df[train_df['Train'] == 0]['Survived'].values.tolist()
#print len(train_label), len(test_label)
train_df.drop('Survived', axis=1, inplace=True)
train_values = train_df.values
# for key in colname[1:]:
#     print 'range', key,  max(train_df[key]) - min(train_df[key])
##learning the data
train_values = scale(train_values)
print train_values.shape
X_train = train_values[train_values[:,-1]>0, :-1]
X_test = train_values[train_values[:,-1]<0, :-1]
print X_train.shape, X_test.shape