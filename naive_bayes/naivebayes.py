
from sklearn import datasets

iris = datasets.load_iris() 

x = iris.data
y = iris.target

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB() 

pre = clf.fit(x,y).predict(x)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(pre, y)

print(accuracy)