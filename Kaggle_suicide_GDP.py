__author__ = "Luke Liu"
#encoding="utf-8"
#研究总GDP 与自杀率之间的关系，以及每个国家的人均GDP与自杀率之间的关系
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
# 国家列表
country_list = data_withHDI.Country.unique()
country_list=sorted(country_list)
# 首先求出suicide_rate
suicide_rate=[]
for country in country_list:
    pop=data[data.Country==country].Population.sum()
    suicide_no=data[data.Country==country].SuicidesNo.sum()
    C_suicide_rate =round(suicide_no/pop,6)
    suicide_rate.append(C_suicide_rate)

#求出每个国家的GDP总量
Total_GDP=[]
for country in country_list:
    min_year=data[data.Country==country].Year.min()
    max_year=data[data.Country==country].Year.max()
    gaps=max_year-min_year+1
    tt_gdp=(data[data.Country==country].GdpForYearMoney.sum())/12
    mt_gdp=tt_gdp/gaps
    Total_GDP.append(mt_gdp)

#求出每个国家的人均GDP
Per_GDP=[]
for country in country_list:
    min_year=data[data.Country==country].Year.min()
    max_year=data[data.Country==country].Year.max()
    pgaps=max_year-min_year+1
    tt_pgdp=(data[data.Country==country].GdpPerCapitalMoney.sum())/12
    mt_pgdp=tt_pgdp/pgaps
    Per_GDP.append(mt_pgdp)
print("----------------------------------")

print(Per_GDP)
print(suicide_rate)

#尝试去分析人均GDP与自杀率之间的关系
dict_A=dict(zip(Total_GDP,suicide_rate))
h =sorted(dict_A.items(),key=lambda x:x[0])
Total_GDP_so=[]
suicide_rate_so_by_Total=[]
for items in h:
    Total_GDP_so.append(items[0])
    suicide_rate_so_by_Total.append(items[1])
# 使用sklearn进行线性拟合
print(Total_GDP_so)
#首先是 Sklearn 进行回归预测分析
from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model
del Total_GDP_so[-1]
del suicide_rate_so_by_Total[-1]
linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(Total_GDP_so).reshape(-1,1)
p2=np.array(suicide_rate_so_by_Total).reshape(-1,1)
min_rate=suicide_rate_so_by_Total[0]
max_rate =suicide_rate_so_by_Total[-1]
linereg01.fit(p1,p2)
plt.ylim(-0.000005,0.0005)
plt.xlim(393909940,856567980317)
y_predict = linereg01.predict(p1)
r=np.random.rand(len(Total_GDP_so))
plt.scatter(Total_GDP_so,suicide_rate_so_by_Total,s=50,c=r)
plt.xlabel("Country's Total GDP ")
plt.ylabel("Suicide Rate")
plt.title("The Relationship between Suicide Rate and Country's Total GDP")
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg01.score(p1,p2)
print("The R^2 Value is {}".format(R_value))
plt.legend()
plt.show()

#Kendall秩相关
from scipy.stats import kendalltau
coef, p = kendalltau(Total_GDP_so, suicide_rate_so_by_Total)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(Total_GDP_so, suicide_rate_so_by_Total)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

from scipy.stats import pearsonr

coef3, p3 = pearsonr(Total_GDP_so, suicide_rate_so_by_Total)
print('Pearsonr correlation coefficient: %.3f' % coef3)
print('Samples are correlated (reject H0) p=%.3f' % p3)

print("此为分隔线\n-------------------------------------------------")

#接下来进行人均GDP与相关性之间的分析与报告
dict_B=dict(zip(Per_GDP,suicide_rate))
h = sorted(dict_B.items(),key=lambda x:x[0])
per_gdp_so=[]
suicide_rate_so_by_per=[]
for n in h:
    per_gdp_so.append(n[0])
    suicide_rate_so_by_per.append(n[1])
print(per_gdp_so)
#首先是 Sklearn 进行回归预测分析
from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model
linereg02 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(per_gdp_so).reshape(-1,1)
p2=np.array(suicide_rate_so_by_per).reshape(-1,1)
linereg02.fit(p1,p2)
plt.ylim(-0.000005,0.0005)
y_predict = linereg01.predict(p1)
r=np.random.rand(len(per_gdp_so))
plt.scatter(per_gdp_so,suicide_rate_so_by_per,s=50,c=r)
plt.xlabel("Country's Average GDP ")
plt.ylabel("Suicide Rate")
plt.title("The Relationship between Suicide Rate and Country's Average GDP")
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg02.score(p1,p2)
print("The R^2 Value is {}".format(R_value))
plt.legend()
plt.show()


#Kendall秩相关
from scipy.stats import kendalltau
coef, p = kendalltau(per_gdp_so, suicide_rate_so_by_per)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(per_gdp_so, suicide_rate_so_by_per)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)



from scipy.stats import pearsonr

coef3, p3 = pearsonr(per_gdp_so, suicide_rate_so_by_per)
print('Pearsonr correlation coefficient: %.3f' % coef3)
print('Samples are correlated (reject H0) p=%.3f' % p3)
