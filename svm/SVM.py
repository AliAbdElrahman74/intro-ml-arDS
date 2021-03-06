
from sklearn import datasets

iris = datasets.load_iris() 

x = iris.data
y = iris.target

from sklearn import svm

clf = svm.SVC(kernel="rbf" , C = 10000)
clf.fit(x,y)

pre = clf.predict(x)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(pre , y)

print(accuracy)