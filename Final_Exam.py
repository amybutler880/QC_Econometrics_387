# -*- coding: utf-8 -*-
"""
Amy Butler
Econ 387
Final Exam
"""

#%% Import the necessary libraries:

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.optimize as optimize
import matplotlib.pyplot as plt
  
#%% Question One Part A:
    
#Create the log-likelihood function:

def LogLikelihood(THETA, y, X, N, T):
    
    kx = X.shape[1]
    B = THETA[:kx]
    PHI = np.exp(THETA[kx])
    SIGMA_SQ_V = np.exp(THETA[kx+1])
    
    #First Term:
    OBS = N*T
    Neg_OBS = -1*OBS 
    Neg_OBS_Divided = Neg_OBS/2
    log_SIGMA_SQ_V = np.log(SIGMA_SQ_V)
    Term1 = Neg_OBS_Divided*log_SIGMA_SQ_V
    
    #Second Term:
    N_Divided = N/2
    log_PHI = np.log(PHI)
    Term2 = N_Divided*log_PHI
    
    #Third Term:
    IN = np.identity(N)
    IT = np.identity(T)
    JT = (1/T)*np.ones((T,T))
    Pu = np.kron(IN, JT)
    Qu = np.kron(IN, IT-JT)
    Term3=(1/(2*SIGMA_SQ_V))*((y-X@B).T@(PHI*Pu+Qu)@(y-X@B))

    #Final log-likelihood:
    Log_Likelihood = Term1+Term2-Term3
    return -1*Log_Likelihood
    
#%%  Question 1 Part B:

def MLE(y, X, N, T):
    K=X.shape[1]
    Initial_Guess = np.concatenate((0.2*np.zeros(K+1), -0.5*np.ones(2)))
    Theta_Hat=optimize.minimize(LogLikelihood,Initial_Guess,args=(y,X,N,T),method="BFGS").x
    betahat = Theta_Hat[:K]
    phihat = np.exp(Theta_Hat[K])
    sig_v2hat = np.exp(Theta_Hat[K+1])
    sigmu2_hat = (sig_v2hat/T)*((1-phihat)/phihat)
    return betahat, phihat, sig_v2hat, sigmu2_hat
 
#%% Question 1 Part C:

#Upload the data set:
Munnel = pd.read_csv(r"C:\Users\aliza\Desktop\munnell.csv", header=0)

#Create the explanatory variables:
Munnel["Variable1"] = np.log(Munnel["P_CAP"]+\
                             Munnel["HWY"]+\
                             Munnel["WATER"]+\
                             Munnel["UTIL"])
Munnel["Variable2"]=np.log(Munnel["PC"])
Munnel["Variable3"]=np.log(Munnel["EMP"])
Munnel["Variable4"]=Munnel["UNEMP"]
Munnel.insert(0,"intercept",1)
                                    
#Set the elplanatory variables:               
Explanatory_Variables = Munnel[["intercept","Variable1", "Variable2", "Variable3", "Variable4"]]
         
#Create the response variable:
Response_Variable = np.log(Munnel["GSP"])

#Cross-sections being analyzed:
N=len(Munnel["STATE"].unique())

#The number of time periods:
T=len(Munnel["YR"].unique())

betahat, phihat, sigvhat, sigmu2_hat = MLE(Response_Variable, Explanatory_Variables, N, T)

print(betahat)
print(phihat)
print(sigvhat)
print(sigmu2_hat)

#%%  Question 2 Data Preparation:

#Upload the csv file:
PCEP = pd.read_csv(r"C:\Users\aliza\Desktop\PCECTPI.csv", header=0)

#Convert "DATE" to a datetime object:
PCEP["DATE"]=pd.to_datetime(PCEP.DATE)

#Extract the years:
PCEP["YEAR"]=PCEP["DATE"].dt.year

#Index the data by year:
PCEP=PCEP.set_index("YEAR")

#Extract the data between 1962 and 2017 to make sure we can compute for 1963:
PCEP_Sliced=PCEP.loc[range(1960,2018)]

#Create the t-1 data column:
PCEP_Sliced["PCECTPI T-1"] = PCEP_Sliced["PCECTPI"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["PCECTPI T-1"] = PCEP_Sliced["PCECTPI T-1"].replace(np.nan,0)

#Set the variables:
PCECTPI_t=PCEP_Sliced["PCECTPI"]
PCECTPI_t1=PCEP_Sliced["PCECTPI T-1"]

#Compute the inflation rate:
PCEP_Sliced["INFL"] = 400*(np.log(PCECTPI_t)-np.log(PCECTPI_t1))

#Delete the "DATE" column:
del PCEP_Sliced["DATE"]

#%% Question 2 Part A:

#Create the INFL t-1 data column:
PCEP_Sliced["INFL T-1"] = PCEP_Sliced["INFL"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["INFL T-1"] = PCEP_Sliced["INFL T-1"].replace(np.nan,0)

#Create the INFL t-2 data column:
PCEP_Sliced["INFL T-2"] = PCEP_Sliced["INFL T-1"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["INFL T-2"] = PCEP_Sliced["INFL T-2"].replace(np.nan,0)

#Create the INFL t-3 data column:
PCEP_Sliced["INFL T-3"] = PCEP_Sliced["INFL T-2"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["INFL T-3"] = PCEP_Sliced["INFL T-3"].replace(np.nan,0)

#Create the INFL t-4 data column:
PCEP_Sliced["INFL T-4"] = PCEP_Sliced["INFL T-3"].shift(1, axis=0)

#Convert NAN's to zeros:
PCEP_Sliced["INFL T-4"] = PCEP_Sliced["INFL T-4"].replace(np.nan,0)

#Create the Delta_INFL_t data column:
PCEP_Sliced["Delta_INFL_t"] = PCEP_Sliced["INFL"]-PCEP_Sliced["INFL T-1"]

#Convert NAN's to zeros:
PCEP_Sliced["Delta_INFL_t"] = PCEP_Sliced["Delta_INFL_t"].replace(np.nan,0)

#Create the Delta_INFL_t1 data column:
PCEP_Sliced["Delta_INFL_t1"] = PCEP_Sliced["INFL T-1"]-PCEP_Sliced["INFL T-2"]

#Convert NAN's to zeros:
PCEP_Sliced["Delta_INFL_t1"] = PCEP_Sliced["Delta_INFL_t1"].replace(np.nan,0)

#Create the Delta_INFL_t2 data column:
PCEP_Sliced["Delta_INFL_t2"] = PCEP_Sliced["INFL T-2"]-PCEP_Sliced["INFL T-3"]

#Convert NAN's to zeros:
PCEP_Sliced["Delta_INFL_t2"] = PCEP_Sliced["Delta_INFL_t2"].replace(np.nan,0)

#Create the Delta_INFL_t3 data column:
PCEP_Sliced["Delta_INFL_t3"] = PCEP_Sliced["INFL T-3"]-PCEP_Sliced["INFL T-4"]

#Convert NAN's to zeros:
PCEP_Sliced["Delta_INFL_t3"] = PCEP_Sliced["Delta_INFL_t3"].replace(np.nan,0)
 
#Set the variables:
INFL_t=PCEP_Sliced["INFL"]
INFL_t1=PCEP_Sliced["INFL T-1"]
INFL_t2=PCEP_Sliced["INFL T-2"]
INFL_t3=PCEP_Sliced["INFL T-3"]
Delta_INFL_t=PCEP_Sliced["Delta_INFL_t"]
Delta_INFL_t1=PCEP_Sliced["Delta_INFL_t1"]
Delta_INFL_t2=PCEP_Sliced["Delta_INFL_t2"]
Delta_INFL_t3=PCEP_Sliced["Delta_INFL_t3"]

#Set the data to be between 1963 and 2017:
INFL_t=INFL_t.loc[range(1963,2018)]
INFL_t1=INFL_t1.loc[range(1963,2018)]
INFL_t2=INFL_t2.loc[range(1963,2018)]
Delta_INFL_t=Delta_INFL_t.loc[range(1963,2018)]
Delta_INFL_t1=Delta_INFL_t1.loc[range(1963,2018)]
Delta_INFL_t1=Delta_INFL_t2.loc[range(1963,2018)]
PCEP_Sliced=PCEP_Sliced.loc[range(1963,2018)]

#Turn all of the series into dataframes:
INFL_t = INFL_t.to_frame()
INFL_t1 = INFL_t1.to_frame()
INFL_t2 = INFL_t2.to_frame()
Delta_INFL_t = Delta_INFL_t.to_frame()
Delta_INFL_t1 = Delta_INFL_t1.to_frame()
Delta_INFL_t2 = Delta_INFL_t2.to_frame()

#Run the model:
Model1 = smf.ols(formula = "Delta_INFL_t ~ INFL_t1 + Delta_INFL_t1 + Delta_INFL_t2", \
                data=PCEP_Sliced).fit()

#Look at the model summary:
print(Model1.summary())

#Augmented Dickey-Fuller Unit Root Test:
    
#Null Hypothesis: The data is time dependent and thus non stationary and a unit root is present
#Alternative Hypothesis: The data is stationary and a unit root is not present

#The Dickey-Fuller statistic is -2.779

#Critical Values:
    # 10%: -2.57
    # 5%:  -2.86
    # 1%:  -3.43
    
# -2.779 < -2.57 : Reject the null at the 10% level
# -2.779 > -2.86 : Fail to reject the null at the 5% level
# -2.779 > -3.43 : Fail to reject the null at the 1% level

#%% Question 2 Part B:

#Create E:
np.random.seed(10)
PCEP_Sliced["epsilon"]=np.random.normal(size=(220,1))
    
#Create variable (alpha)(t) but doing Yt-1+E
PCEP_Sliced["Alpha_t"]=PCEP_Sliced["INFL T-1"]+PCEP_Sliced["epsilon"]

#Set the variable:
Alpha_t=PCEP_Sliced["Alpha_t"]

#Create the data column "trend":
PCEP_Sliced["trend"]=np.arange(1,PCEP_Sliced.shape[0]+1)

#Set the variable "trend":
trend = PCEP_Sliced["trend"]
    
#Run the model:
Model2 = smf.ols(formula = "Delta_INFL_t ~ trend + INFL_t1 +Delta_INFL_t1 + Delta_INFL_t2", \
                data=PCEP_Sliced).fit()

#Look at the model summary:
print(Model2.summary())

#Augmented Dickey-Fuller Unit Root Test:
    
#Null Hypothesis: The data is time dependent and thus non-stationary and a unit root is present
#Alternative Hypothesis: The data is stationary and a unit root is not present

#The Dickey-Fuller statistic is -3.422

#Critical Values for intercept & time trend:
    # 10%: -3.12
    # 5%:  -3.41
    # 1%:  -3.96
    
# -3.422 < -3.12 : Reject the null at the 10% level
# -3.422 < -3.41 : Reject the null at the 5% level
# -3.422 > -3.96 : Fail to reject the null at the 1% level

#Plot INFL:
plt.plot(INFL_t)
plt.title("INFL")

#Based on the graph it appears INFL is stationary
#The falls in line with the idea that we reject the null at the 10% and 5% levels

#%% Question 2 Part C

#Run Model 1:
Model1 = smf.ols(formula = "Delta_INFL_t ~ INFL_t1 + Delta_INFL_t1 + Delta_INFL_t2", \
                data=PCEP_Sliced).fit()

#Summary of the model with an additional lag:
print(Model1.summary())

#INFL_t1 is -2.779 in model one
    
#Run the model from part a with one more lag than model 1:
Model1a = smf.ols(formula = "Delta_INFL_t ~ INFL_t1 + Delta_INFL_t1 + Delta_INFL_t2 + Delta_INFL_t3", \
                data=PCEP_Sliced).fit()
    
#Summary of model 1a:
print(Model1a.summary())

#INFL_t is -2.792 in model 1a

#INFL_t decreases when a lag is added
#INFL_t increases when a lag is taken away
#If we want to reject the null in the Dickey-Fuller test we should add lags

#%% Question 2 Part D

#State whether to accept or reject the null hypothesis based off the adf test in part A
#If adf is greater than the critical values reject the null -- data is stationary -- unit root not present
#If adf is less than the critical values fail to reject the null -- data is non-stationary -- unit root is present

#Augmented Dickey-Fuller Unit Root Test:
    
#Null Hypothesis: The data is time dependent and thus non stationary and a unit root is present
#Alternative Hypothesis: The data is stationary and a unit root is not present

#The Dickey-Fuller statistic is -2.779

#Critical Values:
    # 10%: -2.57
    # 5%:  -2.86
    # 1%:  -3.43
    
# -2.779 < -2.57 : Reject the null at the 10% level
# -2.779 > -2.86 : Fail to reject the null at the 5% level
# -2.779 > -3.43 : Fail to reject the null at the 1% level

#At the 10% level a unit root is not present
#At the 5% and 1% levels a unit root is present

#A failure to reject a null hypothesis does not necessarily mean the null is true
#There just isnt enough evidence to show otherwise so we accept the null as true

#%% Question 2 Part E

#Data duplication for future questions:
Data = PCEP_Sliced.copy()

#Trim the date variable:
DATE = PCEP["DATE"].loc[range(1963,2018)]

#Add the date column to PCEP_Sliced
PCEP_Sliced["DATE"] = DATE

#Trim 15% of the data:
PCEP_Sliced = PCEP_Sliced.loc[range(1971,2009)]

#Drop the index:
PCEP_Sliced.reset_index(drop=True, inplace=True)

#Make the date column the index of PCEP_Sliced:
PCEP_Sliced = PCEP_Sliced.set_index("DATE")

#Quandt Likelihood Ratio Test:
break_tau = pd.date_range(start="1963-01", end="2017-04", freq="QS")
savers = np.empty(len(break_tau))
for i in range(len(break_tau)):
    PCEP_Sliced["D"]=(PCEP_Sliced.index > break_tau[i]).astype(int)
    AR_2 = smf.ols(formula = "Delta_INFL_t ~ Delta_INFL_t1 + Delta_INFL_t2 + D\
                   + D:Delta_INFL_t1 + D:Delta_INFL_t2", \
                   data=PCEP_Sliced).fit(cov_type='HAC', cov_kwds = {'maxlags':8})
    hypothesis = ["Delta_INFL_t1=0", "Delta_INFL_t2=0"]
    ftest = AR_2.f_test(hypothesis)
    savers[i] = ftest.statistic[0][0]
break_tau[np.argmax(savers)]

#%% Question 2 Part F

#Create the validation set with data from 1963 to 2003:
Validation_Data=Data.loc[range(1963,2003)]
    
#Create the test set with data from 2003-01 to 2017-11:
Test_Data=Data.loc[range(2003,2018)]

#Set test variables:
Del_INFL_t=Test_Data["Delta_INFL_t"]
Del_INFL_t1=Test_Data["Delta_INFL_t1"]
Del_INFL_t2=Test_Data["Delta_INFL_t2"]

#Create the AR(2) Model:
AR2 = smf.ols(formula = "Del_INFL_t ~ Del_INFL_t1 + Del_INFL_t2", \
                data=Test_Data).fit()

#Summary of the model:
print(AR2.summary())

#Compute out of sample forecasts:
forecast = AR2.predict(exog=dict(Del_INFL_t=Del_INFL_t))

#Print the forecasts:
print(forecast)

#%% Question 2 Part G

#Find the expectation of the forecasted values:
print(np.mean(forecast))

#The mean for the forecasts is 0.01250797666...
#The mean is super close to zero but is non-zero
#The forecasts are slightly biased

#Find the forecasting errors:
Errors = ((Test_Data["Delta_INFL_t"])*(Test_Data["Delta_INFL_t"])-(forecast*forecast))

#Find the expectation of the forecast errors:
print(np.mean(Errors))

#%% Question 2 Part H

#Find the root mean squared forecast error (RMSFE):
Values = (Test_Data["Delta_INFL_t"]-forecast)
Values = (Values)*(Values)
RMSFE = np.sqrt(np.mean(Values))
print(RMSFE)
#The RMSFE is 1.83919722...
    
#Set validation variables:
Del_INFL_t=Validation_Data["Delta_INFL_t"]
Del_INFL_t1=Validation_Data["Delta_INFL_t1"]
Del_INFL_t2=Validation_Data["Delta_INFL_t2"]

#Create the AR(2) Model:
AR_2 = smf.ols(formula = "Del_INFL_t ~ Del_INFL_t1 + Del_INFL_t2", \
                data=Validation_Data).fit()

#Print the summary of the model:
print(AR_2.summary())

#Compute in sample forecasts:
predictions = AR_2.predict(exog=dict(Del_INFL_t1=Del_INFL_t1))

#Print the predictions:
print(predictions)

#Maniputlate and compare:
residuals = Validation_Data["Delta_INFL_t"] - predictions
residuals = (residuals)*(residuals)
residual_sum = sum(residuals)
comparison_number = residual_sum / ((40*4)-2-1)
print(comparison_number)
#The number is 1.4402798...

#1.83919722... and 1.4402798... are somewhat close but not totally consistent with one another

#%% Question 2 Part I

#There is a big outlier in 2008 Q4
#During that time crude oil prices dropped significantly
#Because crude oil prices dipped inflation fell

#Plot:
plt.plot(INFL_t)
#This graph shows the dip in 2008

    
    

    


