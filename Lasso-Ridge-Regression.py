"""
Linear regression is prone to overfitting which impacts the accuracy of the model on unseen dataset. Therefore to overcome overfitting # TODO:

some extent we use regularization techniques which is basically adding penalty terms to the error function.

Lasso regression : Linear regression with L1 regularisation

Ridge Regression : Linear regression with  L2 regularisation

In L1 regularisation we add lambda*|W| as the penalty term to the error function

In L2 regularisation we add lamdbga*(w)^2 as the penalty term

Reasoning :

L1 regualrisation during gradent desent has an effect of pushing weight towards zero, therfore reducing the number of features

derviative of regularisation term with respect to W  = 1 if W>0 else -1

W_new = (W_old - lambda) + der_Hypothesis (2ax +b) if W > 0
W_new = (W_old + lambda) + der_Hypothesis (2ax +b) if W < 0

L2 regualrisation during gradent desent doesnt necessarilt have an effect of pushing weight towards zero but still shifts weights from perfect value

W_new = (W_old - 2*lambda*w) + der_Hypothesis
"""

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

boston_df['Price']=boston.target

newX=boston_df.drop('Price',axis=1)
print newX[0:3] # check
newY=boston_df['Price']


X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
print len(X_test), len(y_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
rr = Ridge(alpha=0.01)



rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7)
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$')
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()
