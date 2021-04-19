"""
Amy Butler
Problem Set 5
"""
    
#Imports:
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('text', usetex = True)

#Upload cornwell_rupert.csv
data=pd.read_csv(r"C:\Users\aliza\Desktop\cornwell_rupert.csv", header=0)

#Set the index:
data = data.set_index(["ID","YEAR"]) 

#Create and add EXP Squared:
data["EXP_SQ"]= data["EXP"]*data["EXP"]

#%% Question 1:

#a: 
    
#Fixed Effects Regression
regression = PanelOLS.from_formula("LWAGE ~ 1 + EXP + EXP_SQ + WKS + EntityEffects", data)
regression = regression.fit(cov_type = 'clustered', cluster_entity = True)

#Generate a table:
regression.bse = regression.std_errors
regression.tvalues = regression.tstats
regression.model.exog_names = regression._var_names
regression.model.endog_names = "LWAGE"
names = ["(1)"]
some_stats = {"R2": lambda x: "{0:.3f}".format(x.rsquared),\
              "R2_within": lambda x: "{0:.3f}".format(x.rsquared_within),\
              "R2_between": lambda x: "{0:.3f}".format(x.rsquared_between),\
              "No.observations": lambda x: "{0:d}".format(int(x.nobs))}
r_vars = ["EXP", "EXP_SQ", "WKS"]
results_table = summary_col(results=[regression],
                            float_format="%0.3f",
                            stars=True,
                            model_names=names,
                            info_dict=some_stats,
                            regressor_order=r_vars)
results_table = results_table.as_text()
resultFile=open("results.csv", "w")
resultFile.write(results_table)
resultFile.close()
print(results_table)

#The return to an extra year of experience at the sample mean of EXP is 0.114.

#b

#Test for significance:
print(regression.summary)
print(regression.f_pooled)
#The inidividual fixed effects are statistically significant.

#%%

#c

#Random Effects Regression:
random_reg = RandomEffects.from_formula("LWAGE ~ 1 + EXP + EXP_SQ + WKS + ED", data)
random_reg = random_reg.fit(cov_type = 'clustered', cluster_entity = True)

#Generate a table:
random_reg.bse = random_reg.std_errors
random_reg.tvalues = random_reg.tstats
random_reg.model.exog_names = random_reg._var_names
random_reg.model.endog_names = "LWAGE"
names = ["(1)"]
some_stats = {"R2": lambda x: "{0:.3f}".format(x.rsquared),\
              "R2_within": lambda x: "{0:.3f}".format(x.rsquared_within),\
              "R2_between": lambda x: "{0:.3f}".format(x.rsquared_between),\
              "No.observations": lambda x: "{0:d}".format(int(x.nobs))}
r_vars = ["EXP", "EXP_SQ", "WKS", "ED"]
random_reg_table = summary_col(results=[random_reg],
                            float_format="%0.3f",
                            stars=True,
                            model_names=names,
                            info_dict=some_stats,
                            regressor_order=r_vars)
random_reg_table = random_reg_table.as_text()
resultFile=open("random_reg_table.csv", "w")
resultFile.write(random_reg_table)
resultFile.close()
print(random_reg_table)

#d

#Hypothesis Test:
print(random_reg.summary)
#EXP, EXP_SQ, and ED are statistically significant. WKS is not statistically significant. 

#%% Question 2:

#Upload munnell.csv
info=pd.read_csv(r"C:\Users\aliza\Desktop\munnell.csv", header=0)

#Create Variables:
info["lnGSP"]=np.log(info["GSP"])
info["GSP_PC"]=np.log(info["P_CAP"]+info["HWY"]+info["WATER"]+info["UTIL"])
info["lnPC"]=np.log(info["PC"])
info["lnEMP"]=np.log(info["EMP"])
info["lnUNEMP"]=np.log(info["UNEMP"])

#Set the index:
info = info.set_index(["STATE","YR"]) 
    
#a

#Estimate the model:
model = PooledOLS.from_formula("lnGSP ~ 1 + GSP_PC + lnPC + lnEMP + lnUNEMP", info)
model = model.fit(cov_type = 'clustered', cluster_entity = True)

#Generate a table:
model.bse = model.std_errors
model.tvalues = model.tstats
model.model.exog_names = model._var_names
model.model.endog_names = "lnGSP"
names = ["(1)"]
some_stats = {"R2": lambda x: "{0:.3f}".format(x.rsquared),\
              "R2_within": lambda x: "{0:.3f}".format(x.rsquared_within),\
              "R2_between": lambda x: "{0:.3f}".format(x.rsquared_between),\
              "No.observations": lambda x: "{0:d}".format(int(x.nobs))}
r_vars = ["GSP_PC", "lnPC", "lnEMP", "lnUMEP"]
model_table = summary_col(results=[model],
                            float_format="%0.3f",
                            stars=True,
                            model_names=names,
                            info_dict=some_stats,
                            regressor_order=r_vars)
model_table = model_table.as_text()
resultFile=open("model.csv", "w")
resultFile.write(model_table)
resultFile.close()
print(model_table)

#The elasticity of gross state product with respect to public capital is 0.153.
#It is statistically significant at the 5% level.
#The elasticity of gross state product with respect to private capital is 0.308.
#It is statistically significant at the 1% level.

#%%
#b

#Fixed Effects Model:
fixed_model = PanelOLS.from_formula("lnGSP ~ 1 + GSP_PC + lnPC + lnEMP + lnUNEMP + EntityEffects", info)
fixed_model = fixed_model.fit(cov_type = 'clustered', cluster_entity = True)

#Generate a table:
fixed_model.bse = fixed_model.std_errors
fixed_model.tvalues = fixed_model.tstats
fixed_model.model.exog_names = model._var_names
fixed_model.model.endog_names = "lnGSP"
names = ["(1)"]
some_stats = {"R2": lambda x: "{0:.3f}".format(x.rsquared),\
              "R2_within": lambda x: "{0:.3f}".format(x.rsquared_within),\
              "R2_between": lambda x: "{0:.3f}".format(x.rsquared_between),\
              "No.observations": lambda x: "{0:d}".format(int(x.nobs))}
r_vars = ["GSP_PC", "lnPC", "lnEMP", "lnUMEP"]
fixed_model_table = summary_col(results=[fixed_model],
                            float_format="%0.3f",
                            stars=True,
                            model_names=names,
                            info_dict=some_stats,
                            regressor_order=r_vars)
fixed_model_table = fixed_model_table.as_text()
resultFile=open("fixed_model.csv", "w")
resultFile.write(fixed_model_table)
resultFile.close()
print(fixed_model_table)

#The elasticity of gross state product with respect to public capital is -0.023.
#It is not statistically significant.
#The elasticity of gross state product with respect to private capital is 0.291.
#It is statistically significant at the 1% level.

#c

#Hypothesis test:
print(fixed_model.summary)
print(fixed_model.f_pooled)
#The state fixed effects are significant.

#%%
#d

#Random Effects Regression:
random_model = RandomEffects.from_formula("lnGSP ~ 1 + GSP_PC + lnPC + lnEMP + lnUNEMP", info)
random_model = random_model.fit(cov_type = 'clustered', cluster_entity = True)

#Generate a table:
random_model.bse = random_model.std_errors
random_model.tvalues = random_model.tstats
random_model.model.exog_names = random_model._var_names
random_model.model.endog_names = "lnGSP"
names = ["(1)"]
some_stats = {"R2": lambda x: "{0:.3f}".format(x.rsquared),\
              "R2_within": lambda x: "{0:.3f}".format(x.rsquared_within),\
              "R2_between": lambda x: "{0:.3f}".format(x.rsquared_between),\
              "No.observations": lambda x: "{0:d}".format(int(x.nobs))}
r_vars = ["GSP_PC", "lnPC", "lnEMP", "lnUMEP"]
random_model_table = summary_col(results=[random_model],
                            float_format="%0.3f",
                            stars=True,
                            model_names=names,
                            info_dict=some_stats,
                            regressor_order=r_vars)
random_model_table = random_model_table.as_text()
resultFile=open("random_model_table.csv", "w")
resultFile.write(random_model_table)
resultFile.close()
print(random_model_table)

#The elasticity of gross state product with respect to public capital is 0.005.
#It is not statistically significant.
#The elasticity of gross state product with respect to private capital is 0.311.
#It is statistically significant at the 1% level.

#e

#Hypothesis Test:
print(random_model.summary)
#GSP_PC, lnPC, and lnEMP are not statistically significant. lnUNEMP is statistically significant.



    


