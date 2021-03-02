"""
Amy Butler
Problem Set 2
"""

import numpy as np

#%% Question 0

def OLS_Model(X,Y):
    x_bar=np.mean(X)
    y_bar=np.mean(Y)
    n=np.size(X)
    top = np.zeros((n,1))
    bottom = np.zeros((n,1))
    for i in range(0,n-1):
        top[i]=(X[i]-x_bar)*(Y[i]-y_bar)
        bottom[i]=(X[i]-x_bar)**2  
    numerator=sum(top)
    denominator=sum(bottom)
    B1=numerator/denominator
    B0=y_bar-B1*x_bar
    return B1,B0
  
#%% Question 1

np.random.seed(37)

#%% Question 2

tmp1=np.zeros((5000,1))
np.size(tmp1)
I_n=np.ones((1000,1))

for i in range(0,4999):
    X=np.random.normal(size=(1000,1))
    E=np.random.normal(size=(1000,1))
    Y=0.5*I_n+(1.8)*X+E
    B1_Output,B0_Output=OLS_Model(X,Y)
    tmp1[i]=B1_Output
    
#%% Question 3

np.mean(tmp1)
#The mean is 1.799 which is close to 1.8.

#%% Question 4

tmp2=np.zeros((5000,1))
np.size(tmp2)
I=np.ones((1000,1))

for i in range(0,4999):
    x=np.random.normal(size=(1000,1))
    v_i=np.random.normal(size=(1000,1))
    e=-0.5*x+v_i
    Y=0.5*I+(1.8)*x+e
    B1_Output1,B0_Output1=OLS_Model(X,Y)
    tmp2[i]=B1_Output1

#%% Question 5

np.mean(tmp2)
#The mean is further away from 1.8 because the errors are larger so bias occurs and therefore pulls the mean in a different direction.

    

