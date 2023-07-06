import sklearn
from sklearn import svm
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
x=cancer.data
y=cancer.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

model = svm.SVC(kernel="linear", C=1)
model.fit(x_train,y_train)

y_pred= model.predict(x_test)

acc= metrics.accuracy_score(y_test,y_pred)
print(acc)