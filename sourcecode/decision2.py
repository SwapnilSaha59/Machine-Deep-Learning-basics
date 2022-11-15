import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("C:\\Users\\User\\Desktop\\iris_data.csv")

#print(irisData.head)
data.features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets = data.Class 

#with grid search you can find an optimal parameter "parameter tuning"
param_grid = {'max_depth': np.arange(1, 10)}

#in every iteration data is splitted randomly in cross validation + DecisionTreeClassifier
#initializes the tree randomly: thats why you get different results !!!
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets, test_size=.2)

tree.fit(feature_train, target_train)
tree_predictions = tree.predict_proba(feature_test)[:, 1]

print("Best parameter with Grid Search: ", tree.best_params_)

