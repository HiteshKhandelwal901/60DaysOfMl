"""

Algorithm :

Input - data points and its corresponding label. say 2 dimensional data (x1,x2 = y)
1 -> Get the new data point which you want to classify (x1,x2)
2 -> Choose a distance metric, say :  Euclidian distance
3 -> Calculate the distance of the new point to all the points in the dataset. sqrt((x1-x2)^2) + (y1-y2)^2)
4 -> Choose the max k distance and store its corresponding labels
5 -> lebels with highest number of classes will be assigned as the class to the new data point


Features :

Non Parametric :  Since KNN does not make any assumption on the distribution of the data that it is trying to model and does not use fixed number
of parameters to determine the output we call this as Non Parametric Algorithm

Lazy learner :  It does not train but instead does all calcultion during the classification itself

Pros :

- Easy and intutive to implement
- works well with lower dimensiona data sensitive
- No parameters and henc no training needed

Cons :

- Curse of dimesionality, hence works well only with lower dimension
- computationaly expensive as the data grows
- memory is required to store the distances
- Requires the features to be on the same scale

Unknown questions :

How does KNN deal with multicollinearity dataset ?
overlapping dataset ? should work fine
How good is a KNN in regression tasks
Which metric works best for KNN
Best value of K ? Usually taken as K = sqrt(n)

"""
