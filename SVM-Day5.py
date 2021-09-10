"""

SVM - Support vector machine

SVC - Support vector classifier

Say we have a three classification dataset (x1, x2,y)

D1 - first dataset that is linearly separable with no outliers

D2 - Second dataset that is linearly separable with outliers

D3 - Non linear dataset

Now in the case of D1, we find maximimum margin clasifier which means the decison bounray is at maximum distance from both closet point of both class
The distance between the closet point and the decision boundary is called the margin

Now in case of D2, we cant use the above approach because say if there is an outlier closer to the closet point in x2, then the decion boundary will be
between this outlier and the closet x2 point. Therefore the decion bondary is determined by the outlier. Therefore, here svc is sensitive to outliers 
meaning it has high variance. [ see statquest svm theory lecture]

Now in case of D3, we cant use both the above approach because now the data are overlapping and are not lineary seprable. Therefore, here we use a kernel
function and transform the data into higher dimension and then run svc on it. This s called support vector machine. Here, the kernel functin can be linear( first case)
, polynomial(second), RDL etc.

Example : say we have 1D data and we use sqaure function as our kernel polynomial function. Then we take data points in 1d and square it and use it
as Y axis.

"""

import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

#if not linearly separable or consists outliers use kernel as non linear functions
clf = svm.SVC(kernel = 'linear')
clf.fit(X, y)
