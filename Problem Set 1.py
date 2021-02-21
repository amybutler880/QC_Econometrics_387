"""
Amy Butler
Problem Set 1
"""
import numpy as np

#%% Question 0

a1 = np.array([1,2,3,1])
a2 = np.asmatrix(a1)

b1 = np.array(([1],[0],[1],[5]))
b2 = np.asmatrix(b1)

A1 = np.array([[1,3,5],[2,4,6],[7,9,11]])
A2 = np.asmatrix(A1)

B1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
B2 = np.asmatrix(B1)

#%% Question 1

x = np.array([[1,3],[2,4]])
y = np.array([[1,0],[0,1]])
z = np.array([[x,x],[y,y]])

z[0:1,0:1]

#%% Question 2

a2_transpose = np.transpose(a2)
A2_transpose = np.transpose(A2)

(a2)@(a2_transpose)

(b2)+(a2_transpose)

(A2)@(A2_transpose)

(A2)**3

(A2)@(B2)

#%% Question 3

A = np.random.normal(size=(10,5))
B = np.random.normal(size=(5,10))
C = A@B
np.fill_diagonal(C,1)

#%% Question 4

D = np.random.normal(size=(20,15))
E = np.random.normal(size=(15,20))

F=D@E
F[F<=0]=0.5

#%% Question 5

np.reshape(D,(3,10,10))
np.reshape(E,(3,10,10))

#%% Question 6

x = np.arange(100.0)

np.reshape(x,(10,10))
np.reshape(x,(20,5))
np.reshape(x,(10,10,1))

#%% Question 7

x=np.reshape(np.arange(100.0),(5,20))

np.ravel(x)[1:102:2]
np.ndarray.flatten(x)[1:102:2]
x.flat[1:102:2]

#%% Question 8

x=np.array([[16,10],[13,11]])
y=5
z=np.array([[16,10],[13,11],[12,33]])
z_transpose=np.transpose(z)
y_array1=np.tile(y,(2,3))
y_array2=np.tile(y,(1,3))

#Top of the chart:
np.hstack((x,y_array1))
#Bottom of the chart:
np.hstack((z,(np.vstack((z_transpose,y_array2)))))
#The entire chart:
m=np.vstack(((np.hstack((x,y_array1))),(np.hstack((z,(np.vstack((z_transpose,y_array2))))))))
print(m)

np.shape(m) #Shape is (5,5)
np.ndim(m)  #Dimesion is 2
m[2:4, 2:5] #Extracts z_transpose

#%% Question 9

diagonal=np.diag(m)
zero_array=np.zeros((5,5))
np.fill_diagonal(zero_array,diagonal)
print(zero_array)

np.linalg.eig(m)
np.linalg.matrix_rank(m)
np.linalg.det(m)
np.linalg.inv(m)
np.matrix.trace(m)
