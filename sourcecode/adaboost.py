import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

data = pd.read_csv("C:\\Users\\User\\Desktop\\iris_data.csv")

#print(data)
data.features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets = data.Class 

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets, test_size=.2)

model = AdaBoostClassifier(n_estimators=100,learning_rate=1,random_state=123)
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))
