"""
Amy Butler
Problem Set 3
"""
#%% Question 0
#Given AR(2) Model:
import numpy as np
np.random.seed(37)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
T = 200
alpha1 = 0.8
alpha2 = 0.15
u = np.random.randn(T)
y = np.zeros(T)
for t in range(1, T):
 y[t] = alpha1*y[t-1] + alpha2*y[t-2] + u[t]
print(y)
#The for loop starts with one because when dealing with time periods there is no data generated after 0 time has passed.

#Generating 100 Observations Using a While Loop & Plotting:
x = np.zeros(101)
t=1
while t<=100:
 x[t] = alpha1*y[t-1] + alpha2*y[t-2] + u[t]
 t=t+1
print(x[0:100]) #Prints 100 of the 101 outputs (eliminates the zero output)
t_vector=np.arange(99)
plt.plot(t_vector,x[1:100],color="blue", label="y")
plt.title("Simulated AR(2) Process")
plt.xlabel("periods")
plt.ylabel("y")
plt.legend()

#%% Question 1
x=np.arange(15)
odd = [] # empty list for odd numbers
even = [] # empty list for even numbers
for i in x:
 if (i%2==0):
 even=np.append(even,i)
 else:
 odd=np.append(odd,i)
print(even)
print(odd)

#%% Question 2
from string import ascii_lowercase
letters = set(ascii_lowercase)
vowel = set()
consonant = set()
for i in letters:
 if (i=="a" or i=="e" or i=="i" or i=="o" or i=="u"):
 vowel=np.append(vowel,i)
 else:
 consonant=np.append(consonant,i)
print(vowel)
print(consonant)

#%% Question 3
x = np.linspace(0, 10, 100)
sinx = np.sin(x)
cosx = np.cos(x)
plt.plot(x,sinx, label="sin(x)")
plt.plot(x,cosx, linestyle="--", label="cos(x)")
plt.xlabel("x")
plt.ylabel("sin(x)/cos(x)")
plt.legend()
plt.title("Sine and Cosine Functions")

#%% Question 4
x = np.linspace(0, 10, 100)
sinx = np.sin(x)
cosx = np.cos(x)
plt.plot(x,sinx, label="sin(x)", markeredgecolor="black", \
 markersize=5, markerfacecolor="green", marker="o", linewidth=0)
plt.plot(x,cosx, label="cos(x)", markeredgecolor="black", \
 markersize=5, markerfacecolor="red", marker="D", linewidth=0)
plt.xlabel("x")
plt.ylabel("sin(x)/cos(x)")
plt.legend()
plt.title("Sine and Cosine Functions")

#%% Question 5
Languages=["Python","Java","JavaScript","C#","C/C++","PHP","R",\
 "Objective-C","Swift","Matlab","Kotlin","TypeScript",\
 "Go","VBA","Ruby"]
Share=[30.17,17.18,8.21,6.76,6.71,6.13,3.81,3.56,1.82,1.8,\
 1.76,1.74,1.34,1.22,1.13]
plt.barh(np.flip(Languages), np.flip(Share), color="magenta")
plt.title("Popularity of Programming Languages")
plt.xlabel("Share")

#%% Question 6
Languages=["Python","Java","JavaScript","C#","C/C++","PHP","R",\
 "Objective-C","Swift","Matlab","Kotlin","TypeScript",\
 "Go","VBA","Ruby"]
Share=[30.17,17.18,8.21,6.76,6.71,6.13,3.81,3.56,1.82,1.8,\
 1.76,1.74,1.34,1.22,1.13]
plt.pie(Share, labels=Languages, autopct="%1.1f%%")
plt.title("Popularity of Programming Languages")


