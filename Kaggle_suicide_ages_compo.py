__author__ = "Luke Liu"
#encoding="utf-8"
# 读取CSV表格
import  os
import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
csv_file_path= 'master.csv'
data=pd.read_csv(csv_file_path)
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
print("The shape of the data is {}".format(data.shape))
# print(data.isnull().any())
#print(data.isnull().sum())
# save data with HDI
data_withHDI=data[data['HDIForYear']>0]
print(data_withHDI.shape)
# 删除HDI,因为大部分不包含
data=data.drop(['HDIForYear'],axis=1)
data=data.drop(['CountryYear'],axis=1)
#print(data.isnull().sum())
print(data.shape)
#计算不同的年龄组成可能对自杀率产生的影响。
Age_list=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']
people_under_14_po = data[data["Age"]=='5-14 years'].Population.sum()
people_under_24_po = data[data["Age"]=='15-24 years'].Population.sum()
people_under_34_po = data[data["Age"]=='25-34 years'].Population.sum()
people_under_54_po = data[data["Age"]=='35-54 years'].Population.sum()
people_under_74_po = data[data["Age"]=='55-74 years'].Population.sum()
people_above_75_po = data[data["Age"]=='75+ years'].Population.sum()
people_all= data.Population.sum()
people_under_14_rate= people_under_14_po/people_all
people_under_24_rate =people_under_24_po/people_all
people_under_34_rate =people_under_34_po/people_all
people_under_54_rate=people_under_54_po/people_all
people_under_74_rate=people_under_74_po/people_all
people_above_75_rate=people_above_75_po/people_all
people_world_age_dis=[people_under_14_rate,people_under_24_rate,people_under_34_rate,people_under_54_rate,people_under_74_rate,people_above_75_rate]
# print(people_under_14_rate)
# print(people_under_24_rate)
# print(people_under_34_rate)
# print(people_under_54_rate)
# print(people_under_74_rate)
# print(people_above_75_rate)
plt.figure(figsize=(10,5))
sns.barplot(x=people_world_age_dis,y=Age_list,alpha=0.7)
plt.title("The World People Age Distribution")
plt.xlabel("People Age Distribution")
plt.ylabel("Age")
plt.show()
#假设 5-14岁代表1分，15-24岁代表2分，25-34代表3分，34-54代表4分，55-74代表五分，75+代表六分
#每个年龄段乘以所对应的比重，最能够代表此时社会的年龄组成。
#First Let's look at The rate globally
year_list=data.Year.unique()
year_list=sorted(year_list)
P15=[]
P25=[]
P35=[]
P55=[]
P75=[]
P75plus=[]
Year_contri=[]
for year in year_list:

    poplution_of_this_year=data[data.Year==year].Population.sum()
    p15=(data[(data["Year"]==year)&(data["Age"]=='5-14 years')].Population.sum())/poplution_of_this_year
    p25=(data[(data["Year"]==year)&(data["Age"]=='15-24 years')].Population.sum())/poplution_of_this_year
    p35=(data[(data["Year"] == year) & (data["Age"] =='25-34 years')].Population.sum())/poplution_of_this_year
    p55=(data[(data["Year"] == year) & (data["Age"] =='35-54 years')].Population.sum())/poplution_of_this_year
    p75=(data[(data["Year"]==year)&(data["Age"]=='55-74 years')].Population.sum())/poplution_of_this_year
    p75plus=(data[(data["Year"]==year)&(data["Age"]=='75+ years')].Population.sum())/poplution_of_this_year
    P15.append(p15)
    P25.append(p25)
    P35.append(p35)
    P55.append(p55)
    P75.append(p75)
    P75plus.append(p75plus)
    Year_contri.append(p15*1+p25*2+p35*3+p55*4+p75*5+p75plus*6)
print(Year_contri)
print(len(Year_contri))
# compute the suicide rate of the year
suicide_rate=[]
for year in year_list:
    pop = data[data.Year==year].Population.sum()
    suicide_NO = data[data.Year==year].SuicidesNo.sum()
    suicide_rates = suicide_NO / pop
    suicide_rate.append(suicide_rates)
print(len(suicide_rate))
# Next, to sorted the data

combine=dict(zip(Year_contri,suicide_rate))
h=sorted(combine.items(),key=lambda x:x[1])
global_age_so=[]
global_suicide_rate=[]
for items in h:
    global_age_so.append(items[0])
    global_suicide_rate.append(items[1])
print(global_suicide_rate)
# 拟合优度
#首先是 Sklearn 进行回归预测分析
from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model
linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(global_age_so).reshape(-1,1)
p2=np.array(global_suicide_rate).reshape(-1,1)
linereg01.fit(p1,p2)
plt.ylim(0.000075,0.000175)
y_predict = linereg01.predict(p1)
r=np.random.rand(len(global_suicide_rate))
plt.scatter(global_age_so,global_suicide_rate,s=100,c=r)
plt.xlabel("Global Age distribution ")
plt.ylabel("Suicide Rate")
plt.title("The Relationship between global suicide rate and Age distribution")
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg01.score(p1,p2)
print("The R^2 Value is {}".format(R_value))
plt.legend()
plt.show()
#Kendall秩相关
from scipy.stats import kendalltau
coef, p = kendalltau(global_age_so, global_suicide_rate)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(global_age_so, global_suicide_rate)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

from scipy.stats import pearsonr

coef3, p3 = pearsonr(global_age_so, global_suicide_rate)
print('Pearsonr correlation coefficient: %.3f' % coef3)
print('Samples are correlated (reject H0) p=%.3f' % p3)

#所有的国家的名字
country_list=data.Country.unique()
#
def Compute_R(countryname):
     year_list = data[data.Country == countryname].Year.unique()
     year_list=sorted(year_list)
     Age_c_C=[]
     C_suicide=[]
     for year in year_list:
         cpoplution_of_this_year = data[(data.Year == year)&(data.Country==countryname)].Population.sum()
         pc15 = (data[(data["Year"] == year) & (data["Age"] == '5-14 years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         pc25 = (data[(data["Year"] == year) & (data["Age"] == '15-24 years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         pc35 = (data[(data["Year"] == year) & (data["Age"] == '25-34 years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         pc55 = (data[(data["Year"] == year) & (data["Age"] == '35-54 years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         pc75 = (data[(data["Year"] == year) & (data["Age"] == '55-74 years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         pc75plus = (data[(data["Year"] == year) & (data["Age"] == '75+ years')&(data.Country==countryname)].Population.sum()) / cpoplution_of_this_year
         Age_c_C.append(pc15*1+pc25*2+pc35*3+pc55*4+pc75*5+pc75plus*6)
     for year in year_list:
         cpoplution_of_this_year = data[(data.Year == year) & (data.Country==countryname)].Population.sum()
         c_suicide_N=data[(data.Year == year) & (data.Country==countryname)].SuicidesNo.sum()
         c_suicide_Rate=c_suicide_N/cpoplution_of_this_year
         C_suicide.append(c_suicide_Rate)
     # Kendall秩相关
     coef1, p = kendalltau(Age_c_C, C_suicide)
     print('Kendall correlation coefficient: %.3f' % coef1)
     print('Samples are correlated (reject H0) p=%.3f' % p)
     #皮尔森相关分析,Spearman秩相关
     coef2, p = spearmanr(Age_c_C, C_suicide)
     print('Spearmans correlation coefficient: %.3f' % coef2)
     print('Samples are correlated (reject H0) p=%.3f' % p)
     return coef2

R_of_countries=[]
for country in country_list:
    R_of_countries.append(Compute_R(country))
dicts=dict(zip(country_list,R_of_countries))
print(dicts)












