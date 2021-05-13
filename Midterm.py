"""
Amy Butler
Midterm
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer

#Import the data
Sample = pd.read_csv(r"C:\Users\aliza\Desktop\Final_sample.csv", header=0)

#%% Question 1

#Create the Model
def OLS_Model(Data):
    x=Data["female"]
    y=Data["pv1math"]
    regression=smf.ols(formula="y~x", data=Data).fit().params
    return regression

#Group the data by background and run the regression.
mathGenderGap=Sample.groupby("background").apply(OLS_Model)

#Save the slope coefficient estimates.
Slopes=mathGenderGap["x"]

#Find the GGI of each background.
df=Sample[["background","ggi"]]
GGIBackground=df.groupby("background").aggregate(np.mean)

#Create the scatter plot.
plt.scatter(GGIBackground,Slopes, color="black")
plt.xlabel("Gender Gap Index")
plt.ylabel("Math Gender Gap")
plt.ylim((-100,50))
PointLabels=np.unique(Sample["background"])
xlabels=GGIBackground.squeeze()
for i in range(1,35):
    plt.annotate(PointLabels[i],(xlabels[i], Slopes[i]))
m,b=np.polyfit(xlabels, Slopes, 1)
plt.plot(xlabels, m*GGIBackground+b, color="black")
    
#%% Question 2

#Turn the string vraibles into numbers.
Sample['year']=pd.factorize(Sample['year'])[0]
Sample['background']=pd.factorize(Sample['background'])[0]
Sample['country']=pd.factorize(Sample['country'])[0]

#Create the 6 Models
mod1 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + background + country + female*country",
               data=Sample).fit(cov_type="HC3")

mod2 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + background + country + female*country+ lgdppc*female",
               data=Sample).fit(cov_type="HC3")

mod3 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + country + female*country+ lgdppc*female + ggi + lgdppc",
               data=Sample).fit(cov_type="HC3")

mod4 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + background + country + female*country + lgdppc*female + fisced + fisced*female + misced + misced*female",
               data=Sample).fit(cov_type="HC3")

mod5 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + background + country + female*country + lgdppc*female + fisced + fisced*female + misced + misced*female + dadwork + dadwork*female + momwork +momwork*female + homepos +homepos*female",
               data=Sample).fit(cov_type="HC3")

mod6 = smf.ols(formula="pv1math ~ female + ggi*female + age + age*female + diffgrade + diffgrade*female + year + background + country + female*country + lgdppc*female + fisced + fisced*female + misced + misced*female +dadwork +dadwork*female + momwork +momwork*female + homepos +homepos*female + pcgirls + pcgirls*female + private + private*female +metropolis +metropolis*female",
               data=Sample).fit(cov_type="HC3")

#Create the Table:
names = ["(1)","(2)","(3)","(4)","(5)","(6)"]
some_stats={"Observations": lambda x: "{0:d}".format(int(x.nobs)),
            "R2": lambda x: "{0:.3f}".format(x.rsquared_adj)}
r_vars= ["Female", "GGI x Female", "Age of student", "Diff.grade",
         "Diff.grade x Female", "GDP x Female", "Dad educ.", 
         "Dad educ. x Female", "Mom educ.", "Mom educ. x Female",
         "Dad work", "Dad work x Female", "Mom work", "Mom work x Female",
         "Home possessions", "Home posession x Female", 
         "Proportion of girls at school", "Prop. girls x Female", 
         "Private school", "Private school x Female", 
         "School is in a metropolis", "School is in a metropolis x Female",
         "GGI","GDP"]
results_table=summary_col(results=[mod1, mod2, mod3, mod4, mod5, mod6],
                          float_format='%0.3f',
                          stars=True,
                          model_names= names,
                          info_dict=some_stats,
                          regressor_order=r_vars,
                          drop_omitted=False)
results_table.add_title("Table 1-The Gender Equality and the Math Gender Gap")
print(results_table)


