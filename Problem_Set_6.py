# -*- coding: utf-8 -*-
"""
Amy Butler
Problem Set 6
"""

#%% Question 1:
    
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import statsmodels.api as sm

#%% Part I:

#Create a zero array to store the random variables
random_variables0 = np.zeros(100)

#Create a zero array to store the e_labels
e_labels = np.zeros(100)

#Creates an array of 100 random variables
for i in range(0,100,1):
    random_variables0[i]= np.random.normal()

#Creates an array of the e_labels
for j in range(0,100,1):
    e_labels[j]=j 

#Turn e_labels into a data frame
e_frame = pd.DataFrame(e_labels, columns=["e_labels"])

#Turn random_variables into a data frame
random_frame0 = pd.DataFrame(random_variables0, columns=["random_variables"]) 
    
#Put the two data frames into one data frame
data0 = pd.concat([random_frame0, e_frame], axis=1)

#Put the e_labels column first
data0 = data0[["e_labels", "random_variables"]]

#Create a zero array to store the Y_t values:
y_values = np.zeros(100)

#Use a for loop to create an array of the Y_t values
y_values[0]=0 + data0.iloc[0]["random_variables"]
for i in range(1,100,1):
    y_values[i]=y_values[i-1] + data0.iloc[i]["random_variables"]

#Turn y_values into a dataframe
Y_frame = pd.DataFrame(y_values, columns=["Y_values"])

#Put the two data frames into one data frame
data0 = pd.concat([data0, Y_frame], axis=1)

#%% Part II:

#Create a zero array to store the random variables
random_variables1 = np.zeros(100)
#Create a zero array to store the e_labels
a_labels = np.zeros(100)
a = "a"

#Creates an array of 100 random variables
for i in range(0,100,1):
    random_variables1[i]= np.random.normal()

#Creates an array of the e_labels
for j in range(0,100,1):
    a_labels[j]=j 

#Turn e_labels into a data frame
a_frame = pd.DataFrame(a_labels, columns=["a_labels"])

#Turn random_variables into a data frame
random_frame1 = pd.DataFrame(random_variables1, columns=["random_variables"]) 
    
#Put the two data frames into one data frame
data1 = pd.concat([random_frame1, a_frame], axis=1)

#Put the e_labels column first
data1 = data1[["a_labels", "random_variables"]]

#Create a zero array to store the Y_t values:
x_values = np.zeros(100)

#Use a for loop to create an array of the Y_t values
x_values[0]=0 + data1.iloc[0]["random_variables"]
for i in range(1,100,1):
    x_values[i]=x_values[i-1] + data1.iloc[i]["random_variables"]

#Turn y_values into a dataframe
X_frame = pd.DataFrame(x_values, columns=["X_values"])

#Put the two data frames into one data frame
data1 = pd.concat([data1, X_frame], axis=1)

#Combine the two data frames
data = pd.concat([data0, data1], axis=1)

#%% Part III:

#Set the data for the independent and dependent variables:
Y = data["Y_values"]
X = data["X_values"]

#Run the regression:
Regression = smf.ols(formula= "Y ~ 1 + X", data=data).fit()

#Summarize the regression:
print(Regression.summary())

#%% Part A:

#The t-statistic of B1 from part iii is -9.871.
#Because the absolute value of the t-statictic is very far from zero we reject the null hypothesis.
#The R-Squared of the regression was 0.499.

#%% Part B:
    
#Create a zero array to store the t values:
t_values = np.zeros(1000)

#Create a zero array to store the R^2 values:
r_values = np.zeros(1000)

#Run the algorithm 1000 times:
for i in range(0,1000):
    random_variables0 = np.zeros(100) 
    e_labels = np.zeros(100)
    for a in range(0,100,1):
        random_variables0[a]= np.random.normal()
    for b in range(0,100,1):
        e_labels[b]=b
    e_frame = pd.DataFrame(e_labels, columns=["e_labels"])
    random_frame0 = pd.DataFrame(random_variables0, columns=["random_variables"]) 
    data0 = pd.concat([random_frame0, e_frame], axis=1)
    data0 = data0[["e_labels", "random_variables"]]
    y_values = np.zeros(100)
    y_values[0]=0 + data0.iloc[0]["random_variables"]
    for c in range(1,100,1):
        y_values[c]=y_values[c-1] + data0.iloc[c]["random_variables"]
    Y_frame = pd.DataFrame(y_values, columns=["Y_values"])
    data0 = pd.concat([data0, Y_frame], axis=1)
    random_variables1 = np.zeros(100)
    #Create a zero array to store the e_labels
    a_labels = np.zeros(100)
    for d in range(0,100,1):
        random_variables1[d]= np.random.normal()
    for e in range(0,100,1):
        a_labels[e]=e
    a_frame = pd.DataFrame(a_labels, columns=["a_labels"])
    random_frame1 = pd.DataFrame(random_variables1, columns=["random_variables"]) 
    data1 = pd.concat([random_frame1, a_frame], axis=1)
    data1 = data1[["a_labels", "random_variables"]]
    x_values = np.zeros(100)
    x_values[0]=0 + data1.iloc[0]["random_variables"]
    for f in range(1,100,1):
        x_values[f]=x_values[f-1] + data1.iloc[f]["random_variables"]
    X_frame = pd.DataFrame(x_values, columns=["X_values"])
    data1 = pd.concat([data1, X_frame], axis=1)
    data = pd.concat([data0, data1], axis=1)
    Y = data["Y_values"]
    X = data["X_values"]
    Regression = smf.ols(formula= "Y ~ 1 + X", data=data).fit()
    t_values[i]= Regression.tvalues[1] #Save the t values
    r_values[i]=(Regression.rsquared) #Save the r^2 values
    
#Construct histogram 1
plt.figure(1)
t_plot=plt.hist(t_values) 
plt.title("t-values")

#Construct histogram 2
plt.figure(2)
r_plot=plt.hist(r_values)
plt.title("r-values")

#Find the 5th, 50th, and 95th percentiles for the t_values
t_percentiles = np.percentile(t_values, [5,50,95])
print(t_percentiles)

#Find the 5th, 50th, and 95th percentiles for the r_values
r_percentiles = np.percentile(r_values, [5,50,95])
print(r_percentiles)

#Create a zero array to store the t values greater than 1.96:
large_t_values = np.zeros(1000)

#Set a counter
count=0

#Find all of the t values that are greater than 1.96
for i in range(0,1000):
    if (abs(t_values[i]))>1.96:
        count = count+1

#Count is equal to 773. This means that 773 out of 1000 t-values exceed 1.96 in absolute value.

#%% Part C.1:

#Create a zero array to store the t values:
t_values = np.zeros(1000)

#Create a zero array to store the R^2 values:
r_values = np.zeros(1000)

#Run the algorithm 1000 times for T=50:
for i in range(0,1000):
    random_variables0 = np.zeros(50) 
    e_labels = np.zeros(50)
    for a in range(0,50,1):
        random_variables0[a]= np.random.normal()
    for b in range(0,50,1):
        e_labels[b]=b
    e_frame = pd.DataFrame(e_labels, columns=["e_labels"])
    random_frame0 = pd.DataFrame(random_variables0, columns=["random_variables"]) 
    data0 = pd.concat([random_frame0, e_frame], axis=1)
    data0 = data0[["e_labels", "random_variables"]]
    y_values = np.zeros(50)
    y_values[0]=0 + data0.iloc[0]["random_variables"]
    for c in range(1,50,1):
        y_values[c]=y_values[c-1] + data0.iloc[c]["random_variables"]
    Y_frame = pd.DataFrame(y_values, columns=["Y_values"])
    data0 = pd.concat([data0, Y_frame], axis=1)
    random_variables1 = np.zeros(50)
    #Create a zero array to store the e_labels
    a_labels = np.zeros(50)
    for d in range(0,50,1):
        random_variables1[d]= np.random.normal()
    for e in range(0,50,1):
        a_labels[e]=e
    a_frame = pd.DataFrame(a_labels, columns=["a_labels"])
    random_frame1 = pd.DataFrame(random_variables1, columns=["random_variables"]) 
    data1 = pd.concat([random_frame1, a_frame], axis=1)
    data1 = data1[["a_labels", "random_variables"]]
    x_values = np.zeros(50)
    x_values[0]=0 + data1.iloc[0]["random_variables"]
    for f in range(1,50,1):
        x_values[f]=x_values[f-1] + data1.iloc[f]["random_variables"]
    X_frame = pd.DataFrame(x_values, columns=["X_values"])
    data1 = pd.concat([data1, X_frame], axis=1)
    data = pd.concat([data0, data1], axis=1)
    Y = data["Y_values"]
    X = data["X_values"]
    Regression = smf.ols(formula= "Y ~ 1 + X", data=data).fit()
    t_values[i]= Regression.tvalues[1] #Save the t values
    r_values[i]=(Regression.rsquared) #Save the r^2 values
    
#Construct histogram 1
plt.figure(1)
t_plot=plt.hist(t_values) 
plt.title("t-values")

#Construct histogram 2
plt.figure(2)
r_plot=plt.hist(r_values)
plt.title("r-values")

#Find the 5th, 50th, and 95th percentiles for the t_values
t_percentiles = np.percentile(t_values, [5,50,95])
print(t_percentiles)

#Find the 5th, 50th, and 95th percentiles for the r_values
r_percentiles = np.percentile(r_values, [5,50,95])
print(r_percentiles)

#Set a counter
count=0

#Find all of the t values that are greater than 1.96
for i in range(0,1000):
    if (abs(t_values[i]))>1.96:
        count = count+1

#Count is equal to 669. This means that 669 out of 1000 t-values exceed 1.96 in absolute value.

#%% Part C.2:

#Create a zero array to store the t values:
t_values = np.zeros(1000)

#Create a zero array to store the R^2 values:
r_values = np.zeros(1000)

#Run the algorithm 1000 times for T=200:
for i in range(0,1000):
    random_variables0 = np.zeros(200) 
    e_labels = np.zeros(200)
    for a in range(0,200,1):
        random_variables0[a]= np.random.normal()
    for b in range(0,200,1):
        e_labels[b]=b
    e_frame = pd.DataFrame(e_labels, columns=["e_labels"])
    random_frame0 = pd.DataFrame(random_variables0, columns=["random_variables"]) 
    data0 = pd.concat([random_frame0, e_frame], axis=1)
    data0 = data0[["e_labels", "random_variables"]]
    y_values = np.zeros(200)
    y_values[0]=0 + data0.iloc[0]["random_variables"]
    for c in range(1,200,1):
        y_values[c]=y_values[c-1] + data0.iloc[c]["random_variables"]
    Y_frame = pd.DataFrame(y_values, columns=["Y_values"])
    data0 = pd.concat([data0, Y_frame], axis=1)
    random_variables1 = np.zeros(200)
    #Create a zero array to store the e_labels
    a_labels = np.zeros(200)
    for d in range(0,200,1):
        random_variables1[d]= np.random.normal()
    for e in range(0,200,1):
        a_labels[e]=e
    a_frame = pd.DataFrame(a_labels, columns=["a_labels"])
    random_frame1 = pd.DataFrame(random_variables1, columns=["random_variables"]) 
    data1 = pd.concat([random_frame1, a_frame], axis=1)
    data1 = data1[["a_labels", "random_variables"]]
    x_values = np.zeros(200)
    x_values[0]=0 + data1.iloc[0]["random_variables"]
    for f in range(1,200,1):
        x_values[f]=x_values[f-1] + data1.iloc[f]["random_variables"]
    X_frame = pd.DataFrame(x_values, columns=["X_values"])
    data1 = pd.concat([data1, X_frame], axis=1)
    data = pd.concat([data0, data1], axis=1)
    Y = data["Y_values"]
    X = data["X_values"]
    Regression = smf.ols(formula= "Y ~ 1 + X", data=data).fit()
    t_values[i]= Regression.tvalues[1] #Save the t values
    r_values[i]=(Regression.rsquared) #Save the r^2 values
    
#Construct histogram 1
plt.figure(1)
t_plot=plt.hist(t_values) 
plt.title("t-values")

#Construct histogram 2
plt.figure(2)
r_plot=plt.hist(r_values)
plt.title("r-values")

#Find the 5th, 50th, and 95th percentiles for the t_values
t_percentiles = np.percentile(t_values, [5,50,95])
print(t_percentiles)

#Find the 5th, 50th, and 95th percentiles for the r_values
r_percentiles = np.percentile(r_values, [5,50,95])
print(r_percentiles)

#Set a counter
count=0

#Find all of the t values that are greater than 1.96
for i in range(0,1000):
    if (abs(t_values[i]))>1.96:
        count = count+1

#Count is equal to 825. This means that 825 out of 1000 t-values exceed 1.96 in absolute value.

#As the sample size increases it seems as though the fraction is approaching 100%.

#%% Question 2:

#Upload the csv file:
PCEP = pd.read_csv(r"C:\Users\aliza\Desktop\PCECTPI.csv", header=0)

#Convert "DATE" to a datetime object:
PCEP["DATE"]=pd.to_datetime(PCEP.DATE)

#Extract the years:
PCEP["YEAR"]=PCEP["DATE"].dt.year

#Index the data by year:
PCEP=PCEP.set_index("YEAR")

#Extract the data between 1962 and 2017 to make sure we can compute for 1963:
PCEP_Sliced=PCEP.loc[range(1962,2018)]

#%% Part A:

#Create the t-1 data column:
PCEP_Sliced["PCECTPI T-1"] = PCEP_Sliced["PCECTPI"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["PCECTPI T-1"] = PCEP_Sliced["PCECTPI T-1"].replace(np.nan,0)

#Set the variables:
PCECTPI_t = PCEP_Sliced["PCECTPI"]
PCECTPI_t1 = PCEP_Sliced["PCECTPI T-1"]

#Compute the inflation rate:
PCEP_Sliced["INFL"] = 400*(np.log(PCECTPI_t)-np.log(PCECTPI_t1))

#The inflation values represent percent change in average price.

#%% Part B:

#Set the variable:
INFL = PCEP_Sliced["INFL"]

#Make INFL a data frame:
INFL = INFL.to_frame()

#Set the INFL data to be between 1963 and 2017:
INFL=INFL.loc[range(1963,2018)]

#Plot the INFL data:
INFL.plot()

#The INFL has a stochastic trend because it is random and varies over time.

#Set the INFL data back to the years 1962 to 2017 for further analysis:
INFL=PCEP_Sliced["INFL"]

#%% Part C:
    
#Compute new variables to use in the four autocorrelations:
PCEP_Sliced["PCECTPI T-2"] = PCEP_Sliced["PCECTPI"].shift(2, axis=0)
PCEP_Sliced["PCECTPI T-3"] = PCEP_Sliced["PCECTPI"].shift(3, axis=0)
PCEP_Sliced["PCECTPI T-4"] = PCEP_Sliced["PCECTPI"].shift(4, axis=0)

#Set the variables:
PCECTPI_t=PCEP_Sliced["PCECTPI"]
PCECTPI_t1=PCEP_Sliced["PCECTPI T-1"]
PCECTPI_t2=PCEP_Sliced["PCECTPI T-2"]
PCECTPI_t3=PCEP_Sliced["PCECTPI T-3"]
PCECTPI_t4=PCEP_Sliced["PCECTPI T-4"]

#Turn all of the series into dataframes:
PCECTPI_t = PCECTPI_t.to_frame()
PCECTPI_t1 = PCECTPI_t1.to_frame()
PCECTPI_t2 = PCECTPI_t2.to_frame()
PCECTPI_t3 = PCECTPI_t3.to_frame()
PCECTPI_t4 = PCECTPI_t4.to_frame()

#Extract all the data from 1963 to 2017:
INFL=INFL.loc[range(1963,2018)]
PCECTPI_t=PCECTPI_t.loc[range(1963,2018)]
PCECTPI_t1=PCECTPI_t1.loc[range(1963,2018)]
PCECTPI_t2=PCECTPI_t2.loc[range(1963,2018)]
PCECTPI_t3=PCECTPI_t3.loc[range(1963,2018)]
PCECTPI_t4=PCECTPI_t4.loc[range(1963,2018)]
PCEP_Sliced=PCEP_Sliced.loc[range(1963,2018)]
    
#Compute the first four autocorrelations of INFL:
AR_1 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})

AR_2 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1 + PCECTPI_t2',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})

AR_3 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1 + PCECTPI_t2 + PCECTPI_t3',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})

AR_4 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1 + PCECTPI_t2 + PCECTPI_t3 + PCECTPI_t4',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})

#Summarize each autocorrelation:
print(AR_1.summary())
print(AR_2.summary())
print(AR_3.summary())
print(AR_4.summary())

#%% Part D:

#Plot the different values of INFL based on time:
plt.figure(1)
plt.scatter(PCECTPI_t,INFL)
plt.scatter(PCECTPI_t1,INFL)
plt.scatter(PCECTPI_t2,INFL)
plt.scatter(PCECTPI_t3,INFL)
plt.scatter(PCECTPI_t4,INFL)

#The behavior of the plot is consistent with this first autocorrelation.
#The data is random and varies over time, making it stochastic.

#%% Part E:

#Run AR(1):
AR_1 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})
print(AR_1.summary())

#Knowing the change in inflation does help predict the change in the inflation.
#As seen in point on the plot above there is a direct correlation from one time period to the next.
#The adjusted R^2 value is 0.843 which shows a fairly high correlation.

#%% Part F:

#Run AR(2):
AR_2 = smf.ols(formula = 'INFL ~ PCECTPI_t + PCECTPI_t1 + PCECTPI_t2',
               data = PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})
print(AR_2.summary())

#Knowing the change in inflation does help predict the change in the inflation.
#The R^2 value is 0.831 which shows a fairly high correlation.

#%% Part G:

#Estimate an AR(p) model and find AIC & BIC:
lag_order = sm.tsa.arma_order_select_ic(y=INFL,
                                        max_ar = 8, max_ma = 0,
                                        ic = ['aic', 'bic'],
                                        trend = 'nc')
print('The lag order selected by AIC is {0:1.0f} and by BIC is {1:1.0f}.'
.format(lag_order.aic_min_order[0], lag_order.bic_min_order[0]))

#The lag order selected by AIC is 3
#The lag order selceted by BIC is 3

#%% Part H

#Get the data from 2017:
PCEP_Sliced_Subset=PCEP_Sliced.loc[range(2017,2018)]

#Set the variables:
INFL_Subset = PCEP_Sliced_Subset["INFL"]
PCECTPI_t_Subset = PCEP_Sliced_Subset["PCECTPI"]
PCECTPI_t1_Subset = PCEP_Sliced_Subset["PCECTPI T-1"]
PCECTPI_t2_Subset = PCEP_Sliced_Subset["PCECTPI T-2"]

#Turn all of the series into dataframes:
PCECTPI_t_Subset = PCECTPI_t_Subset.to_frame()
PCECTPI_t1_Subset = PCECTPI_t1_Subset.to_frame()
PCECTPI_t2_Subset = PCECTPI_t2_Subset.to_frame()

#Run the AR(2) model:
AR_2 = smf.ols(formula = 'INFL_Subset ~ PCECTPI_t_Subset + PCECTPI_t1_Subset + PCECTPI_t2_Subset',
               data = PCEP_Sliced_Subset).fit(cov_type='HAC', cov_kwds = {'maxlags':8})
print(AR_2.summary())







    
    




    
