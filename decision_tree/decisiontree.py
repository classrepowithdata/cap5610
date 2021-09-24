#! /usr/bin/python

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pydotplus
import io

# load data
train_set = pd.read_csv("train.csv")

# preprocess
train_set = train_set.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
train_set = train_set.dropna()

target = train_set.Survived
source = train_set[['Sex', 'Pclass', 'Age']]

# reevaluate some things
source.loc[source.Sex=='female', 'Sex']=1
source.loc[source.Sex=='male', 'Sex']=0
source["Sex"] = source["Sex"].astype(str).astype(float)



# train using 80% of the dataset, check with 20%
x_train, x_test, y_train, y_test = train_test_split(source, target, test_size=0.8, random_state=1)

classif = DecisionTreeClassifier()
classif = classif.fit(x_train, y_train)

y_pred = classif.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = io.StringIO()

export_graphviz(classif, out_file=dot_data, filled=True, rounded=True,
    special_characters=True, feature_names=['Sex', 'Pclass', 'Age'], class_names=['0', '1'])
 

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png("decision_tree.png")