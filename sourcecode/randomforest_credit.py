import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!
# 84% simple kNN without normalizing the dataset
# we can achieve ~ 99% with random forests

credit_data = pd.read_csv("C:\\Users\\User\\Desktop\\credit_data.csv")

#print(credit_data.head())
#print(credit_data.describe())
#print(credit_data.corr())

features = credit_data[["income","age","loan"]]
targets = credit_data.default

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)

model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))