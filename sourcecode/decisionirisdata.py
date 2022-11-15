import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

data = pd.read_csv("C:\\Users\\User\\Desktop\\iris_data.csv")

print(data)
data.features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets = data.Class 

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets, test_size=.2)

model = DecisionTreeClassifier(criterion='gini')
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

predicted = cross_validation.cross_val_predict(model,data.features,data.targets, cv=10)
print(accuracy_score(data.targets,predicted))