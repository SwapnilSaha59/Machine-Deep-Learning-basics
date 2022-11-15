import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\User\\Desktop\\credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!
# 84%

#print(creditData.head())
#print(creditData.describe())
#print(creditData.corr())

data.features = data[["income","age","loan"]]
data.target = data.default

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.target, test_size=0.3)

model = GaussianNB()  
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))