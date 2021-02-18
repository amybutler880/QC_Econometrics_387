"""
Amy Butler
Problem Set 0
"""
#%% Question 0

#a
x = 2
type(x)
#int

#b
x = 2.0
type(x)
#float

#c
x = "2.0"
type(x)
#str

#d
x = 2+0j
type(x)
#complex

#e
x = "2+0j"
type(x)
#str

#f
import math
math.pi
x = math.pi
type(x)
#float


#%% Question 1

s = "Python has a nice syntax."

#a
len(s)
#25

#b
s[18:26]
#"syntax"

#c
print(s[18:26].count("a"))
#1

#d
print(s.replace("a","A"))
#Python hAs A nice syntAx.

#e
t = " But, Stata's is counterintuitive."
print(s + t)
#Python has a nice syntax. But, Stata's is counterintuitive.

#%% Question 2

y = [[1,math.pi,0,7,8],["abc",bool(0),-3,12]]

#a
y[0]
y[0][::-1]
#[8,7,0,3.145...,1]

#b
y[1][1]=bool(1)
print(y[1])
#["abc",True,-3,12]

#c
y[0].remove(math.pi)
print(y[0])
#[1,0,7,8]

#d
y.append([math.pi,math.e,3-1j,False])
print(y)
#[[1,0,7,8],["abc",True,-3,12],[3.14,2.7,3-1j,False]]

#%% Question 3

z = y

#a
z[2][0].replace(math.pi,True)
#"Float object has no attribute replace"

#b
z[2][0]=True
print(z)
#[[1,0,7,8],["abc",True,-3,12],[True,2.7,3-1j,False]]

#%% Question 4

Student = {"Name":"Jane","Age":22,"Courses": ["MATH131","ANTH201"],"Phone":"555-788-5544"}

#a
Student["Phone"]
#555-788-5544

#b
Student["Age"]=24
Student["Age"]
Student["Courses"]=["MATH131","ANTH201","ECO387"]
Student["Courses"]
#24 & ["MATH131","ANTH201","ECO387"]

#c
del Student["Phone"]
print(Student)
#{Name:Jane, Age:24, Courses:{Math131,ANTH201,ECO387}}

#d
Courses = "MATH131-ANTH201-ECO387"
print(Courses)
#MATH131-ANTH201-ECO387

#%% Question 5

w1=set()

#a
w1=set(range(2,102,2))
print(w1)
#w1={2,4,6,...100}

#b
w2=set(range(-100,102,2)) 
sorted(w2)    
print(w2)
#w2={-100,-98,...,98,100}

#c
w1.intersection(w2)
#{2,4,6,...,100}

#d
d = w2-w1
print(d)
#{-100,-98,-96,...,0}

