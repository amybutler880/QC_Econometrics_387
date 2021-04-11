"""
Amy Butler
Problem Set 4
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer

#%% Question 0:

#Upload the csv file:
mort_data=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\mortality_data.csv", header=0)  

#Index the data frame by year:
index_mort_data=mort_data.set_index("year")

#Extract the data between 1993 and 2015:
sliced_mort_data=index_mort_data.loc[range(1993,2016)]
                    
#%% Question 1:

#Turn the index label into data columns:
mort_rename=sliced_mort_data.rename(columns={"mx":"mort_rate",
                                             "qx":"prob_death",
                                             "ax":"ave_length_surv",
                                             "lx":"num_of_surv",
                                             "dx":"num_of_deaths",
                                             "Lx":"num_years_lived",
                                             "Tx":"num_years_left",
                                             "ex":"life_expec"})

#%% Question 2:

#Seperate the age column:
seperate_mort_by_dash=mort_rename["age"].str.partition("-")

seperate_mort_by_plus=seperate_mort_by_dash[0].str.partition("+")

#Add the first column of seperate_mort_by_plus to mort_rename:
mort_rename["age2"]=seperate_mort_by_plus[0]

#Make age2 numeric:
mort_rename[["age2"]]=mort_rename[["age2"]].apply(pd.to_numeric) 

#%% Question 3:

#Create ageGroup:
cut_mort_rename=pd.cut(
    mort_rename["age2"],
    [-1,18,64,300],
    labels=["<18","18-64",">64"])

#Add ageGroup to mort_rename:
mort_rename["ageGroup"]=cut_mort_rename

#%% Question 4:
    
#Drop the columns age and age2:
mort_rename_drop=mort_rename.drop(["age","age2"],axis=1)

#Remove the year index:
mort_rename_drop.reset_index(inplace=True)

#Reorder the columns:
reordered_mort=mort_rename_drop[["state","year","ageGroup","mort_rate",
                                 "prob_death","ave_length_surv","num_of_surv",
                                "num_of_deaths","num_years_lived",
                                "num_years_left","life_expec"]]

#%% Question 5:

#aggregate the mort data:
aggregate_mort=reordered_mort.groupby(["state","year","ageGroup"],dropna=True).sum()

#%% Question 6 & 7:

#Upload the csv file:
inc_data=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\income_data.csv", header=0) 

#Convert to long form and sort by state & year:
inc_data=pd.wide_to_long(inc_data,stubnames=['pinc'],i='state',j='year',sep=".")

#%% Question 8:

#Upload education data from 1993-2006:
ed93=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1993.csv", header=0)
ed94=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1994.csv", header=0)
ed95=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1995.csv", header=0)
ed96=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1996.csv", header=0)
ed97=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1997.csv", header=0)
ed98=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1998.csv", header=0)
ed99=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_1999.csv", header=0)
ed00=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2000.csv", header=0)
ed01=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2001.csv", header=0)
ed02=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2002.csv", header=0)
ed03=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2003.csv", header=0)
ed04=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2004.csv", header=0)
ed05=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2005.csv", header=0)
ed06=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education\education_2006.csv", header=0)

#Upload education_0715:
ed0715=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\education_0715.csv", header=0) 

#The dataset for 2004 has a typo, fix it:
ed04.rename(columns={"Pprcent_college":"percent_college"},inplace=True)

#Create educ_data:
dataframes=[ed93,ed94,ed95,ed96,ed97,ed98,ed99,ed00,ed01,ed02,ed03,ed04,ed05,ed06,ed0715]
educ_data=pd.concat(dataframes,ignore_index=True)

#Rename columns:
educ_data.rename(columns={"percent_highschool":"phs","percent_college":"pcoll"},inplace=True)

#The dataset has rows called footnotes by accident, remove those:
educ_data=educ_data.drop(labels=[550,601,652,703])

#%% Question 9:

#Upload the expenditure datasets from 1993-2015:
ex93=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1993.csv", header=0)
ex94=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1994.csv", header=0)
ex95=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1995.csv", header=0)
ex96=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1996.csv", header=0)
ex97=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1997.csv", header=0)
ex98=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1998.csv", header=0)
ex99=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_1999.csv", header=0)
ex00=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2000.csv", header=0)
ex01=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2001.csv", header=0)
ex02=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2002.csv", header=0)
ex03=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2003.csv", header=0)
ex04=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2004.csv", header=0)
ex05=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2005.csv", header=0)
ex06=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2006.csv", header=0)
ex07=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2007.csv", header=0)
ex08=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2008.csv", header=0)
ex09=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2009.csv", header=0)
ex10=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2010.csv", header=0)
ex11=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2011.csv", header=0)
ex12=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2012.csv", header=0)
ex13=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2013.csv", header=0)
ex14=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2014.csv", header=0)
ex15=pd.read_csv(r"C:\Users\aliza\Desktop\state_data\expenditure\expnd_2015.csv", header=0)
    
#Fix the column names that are incorrect:
ex93.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex94.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex95.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex96.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex97.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex98.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex99.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex00.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex01.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex02.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex03.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex04.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex05.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex06.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex07.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex08.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex09.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex10.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex11.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex12.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex13.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex14.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]
ex15.columns=["state","year","Total_revenue","Taxes","Total_expenditure",\
             "Education","Public_welfare","Hospitals","Health"]

#Create expnd_data:
frames=[ex93,ex94,ex95,ex96,ex97,ex98,ex99,ex00,ex01,ex02,ex03,ex04,ex05,ex06,\
      ex07,ex08,ex09,ex10,ex11,ex12,ex13,ex14,ex15]
expnd_data=pd.concat(frames,ignore_index=True)

#%% Question 10:

#Merge inc_data and educ_data:
data=inc_data.merge(educ_data,on=['state','year'])

#%% Question 11:

#Merge data and expnd_data:
data=data.merge(expnd_data,on=['state','year'])

#%% Question 12:

#Merge mort_data and data:
data=reordered_mort.merge(data,on=['state','year'])

#%% Question 13:

#Remove dataframes (my mort_data is called mort_reordered):
del reordered_mort, inc_data, educ_data, expnd_data

#%% Question 14:

#Remove the commas from numeric values:
data=data.replace(",","",regex=True)

#Change the variables to numeric:
data[["pinc"]]=data[["pinc"]].apply(pd.to_numeric) 
data[["Total_revenue"]]=data[["Total_revenue"]].apply(pd.to_numeric) 
data[["Taxes"]]=data[["Taxes"]].apply(pd.to_numeric) 
data[["Total_expenditure"]]=data[["Total_expenditure"]].apply(pd.to_numeric) 
data[["Education"]]=data[["Education"]].apply(pd.to_numeric) 
data[["Public_welfare"]]=data[["Public_welfare"]].apply(pd.to_numeric) 
data[["Hospitals"]]=data[["Hospitals"]].apply(pd.to_numeric) 
data[["Health"]]=data[["Health"]].apply(pd.to_numeric) 

#Change the measurements:
data["pinc"]/=1e4
data["Total_revenue"]/1e4
data["Taxes"]/=1e4
data["Total_expenditure"]/=1e4
data["Education"]/=1e4
data["Public_welfare"]/=1e4
data["Hospitals"]/=1e4
data["Health"]/=1e4

#%% Question 15:

#Change the measurments of phs and pcoll:
data["phs"]/=100
data["pcoll"]/=100

#%% Question 16:

#Generate a table of descriptive statistics:
data_table=data.describe()

#%% Question 17:

#Change "pinc" to "log_pinc":
data["pinc"]=np.log(data["pinc"])

#Extract data for the age group ">64":
index_data=data.set_index("ageGroup")
sliced_data=index_data.loc[">64"]
    
#Regress:
spec1 = smf.ols(formula="mort_rate ~ Health + Hospitals + pinc + phs\
               + pcoll",data=sliced_data).fit(cov_type="HC3")
               
#%% Question 18:

#Factorize state:
data['state']=pd.factorize(data['state'])[0]
    
#Regress:
spec2 = smf.ols(formula="mort_rate ~ Health + Hospitals + pinc + phs\
               + pcoll +state",data=sliced_data).fit(cov_type="HC3")
               
#%% Question 19:

#Factorize year:
data['year']=pd.factorize(data['year'])[0]
    
#Regress:
spec3 = smf.ols(formula="mort_rate ~ Health + Hospitals + pinc + phs\
               + pcoll +state +year",data=sliced_data).fit(cov_type="HC3")

#%% Question 20:

#Summarize the regressions:
regression_table=summary_col([spec1,spec2,spec3])
print(regression_table)
            


