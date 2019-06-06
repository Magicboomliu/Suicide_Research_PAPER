__author__ = "Luke Liu"
#encoding="utf-8"

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
# 国家列表
country_list = data_withHDI.Country.unique()
country_list=sorted(country_list)
each_country_and_HDI=[]
for country in country_list:
    mean_HDI = round(data_withHDI[data_withHDI.Country==country].HDIForYear.mean(),6)
    each_country_and_HDI.append(mean_HDI)
print(each_country_and_HDI)

suicide_rate=[]
for country in country_list:
    pop = data_withHDI[data_withHDI["Country"]==country].Population.sum()
    suicide_no_c = data_withHDI[data_withHDI["Country"]==country].SuicidesNo.sum()
    C_suicide_rate = round(suicide_no_c/pop,6)
    suicide_rate.append(C_suicide_rate)
print(suicide_rate)

dicts=dict(zip(each_country_and_HDI,suicide_rate))
h = sorted(dicts.items(),key=lambda x:x[0])
suicide_rate_so=[]
each_country_and_HDI_so=[]
for items in h:
    each_country_and_HDI_so.append(items[0])
    suicide_rate_so.append(items[1])

#现在开始线性拟合
#首先是 Sklearn 进行回归预测分析
from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model
linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(each_country_and_HDI_so).reshape(-1,1)
p2=np.array(suicide_rate_so).reshape(-1,1)
linereg01.fit(p1,p2)
plt.ylim(0,0.000175)
y_predict = linereg01.predict(p1)
r=np.random.rand(len(each_country_and_HDI_so))
plt.scatter(each_country_and_HDI_so,suicide_rate_so,s=40,c=r)
plt.xlabel("Different Country's HDI")
plt.ylabel("Suicide Rate")
plt.title("The Relationship between HDI and Suicide Rate")
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg01.score(p1,p2)
print("The R^2 Value is {}".format(R_value))
plt.legend()
plt.show()

#Kendall秩相关
from scipy.stats import kendalltau
coef, p = kendalltau(each_country_and_HDI_so, suicide_rate_so)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(each_country_and_HDI_so, suicide_rate_so)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

from scipy.stats import pearsonr

coef3, p3 = pearsonr(each_country_and_HDI_so, suicide_rate_so)
print('Pearsonr correlation coefficient: %.3f' % coef3)
print('Samples are correlated (reject H0) p=%.3f' % p3)

