__author__ = "Luke Liu"
#encoding="utf-8"

import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
csv_file_path= 'master.csv'
data=pd.read_csv(csv_file_path)
# 输出前五行的相关信息
#print(data.head())
# 输出最后五行的相关信息
#print(data.tail())
#随机显示5行
#data.sample(5)
#随机10%
#data.sample(frac=0.1)
#Describe function includes analysis of all our numerical data. For this, count, mean, std, min,% 25,% 50,% 75, max values are given.
#data.describe()
#当前data的信息
# data.info()
#data的列信息
# print(data.columns)
#给每一列的信息更改名字
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


#世界自杀率变化图
year_list=data.Year.unique()
year_list=sorted(year_list)
suicide_rate=[]
for year in year_list:
    pop_sum=data[data.Year==year].Population.sum()
    suicide_no=data[data.Year==year].SuicidesNo.sum()
    suicide_rate_year=round(suicide_no/pop_sum,7)
    suicide_rate.append(suicide_rate_year)



#人口增长速率
increase=[]
for year in year_list:
    pop=data[data.Year==year].Population.sum()
    if year==1985:
        increase.append(0)
    else:
        pop_prevoius=data[data.Year==year-1].Population.sum()
        increase.append((pop-pop_prevoius)/pop)
sum=0
for i in increase:
    sum+=i
increase[0]=0
print(suicide_rate)
print(increase)

dicta=dict(zip(increase,suicide_rate))
h=sorted(dicta.items(),key=lambda x:x[0])
increase_so=[]
suicide_rate_so=[]

for item in h:
    increase_so.append(item[0])
    suicide_rate_so.append(item[1])
del increase_so[0]
del suicide_rate_so[0]
print(increase_so)
print(suicide_rate_so)


def min_and_max(lst):
    ls=sorted(lst)
    min=ls[0]
    rs=sorted(lst,reverse=True)
    max=rs[0]

    return (min,max)

#线性拟合

from  sklearn import model_selection
from sklearn import  metrics

from sklearn import linear_model

linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(increase_so).reshape(-1,1)
p2=np.array(suicide_rate_so).reshape(-1,1)

linereg01.fit(p1,p2)
y_predict = linereg01.predict(p1)
plt.ylim(min_and_max(suicide_rate_so)[0],min_and_max(suicide_rate_so)[1])
r=np.random.rand(len(increase_so))
plt.xlabel("Population Increase Rate")
plt.ylabel("Suicide Rate")
plt.scatter(increase_so,suicide_rate_so,c='r',s=60,label='orginal_data') #画出散点图
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg01.score(p1,p2)
print("The R^2 is {}".format(R_value))
plt.legend()
plt.show()

from scipy.stats import kendalltau
coef, p = kendalltau(increase_so, suicide_rate_so)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)
#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(increase_so, suicide_rate_so)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

from scipy.stats import pearsonr
coef, p = pearsonr(increase_so, suicide_rate_so)
print('Pearsonr correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

country_list=data.Country.unique()
def Compute_country(countryname):
    #自杀率变化图
    suicide_rate_C = []
    for year in year_list:
        if year>1986 and year<2011:
            pop_sum_c = data[(data.Year == year)&(data.Country==countryname)].Population.sum()
            suicide_no_c = data[(data.Year == year)&(data.Country==countryname)].SuicidesNo.sum()
            suicide_rate_year_c = round(suicide_no_c / pop_sum_c, 7)
            suicide_rate_C.append(suicide_rate_year_c)
    # 人口增长速率
    increase_c = []
    year_listq=sorted(year_list)
    for year in year_listq:
        if year > 1986 and year < 2011:
            pop_c= data[(data.Year == year)&(data.Country==countryname)].Population.sum()
            if year == 1987:
                increase_c.append(0)
            else:
                if pop_c==0:
                    pop_c
                pop_prevoius_c = data[(data.Year == year-1)&(data.Country==countryname)].Population.sum()
                increase_c.append((pop_c - pop_prevoius_c) / pop_c)
    print(increase_c)
    print(".......")
    print(suicide_rate_C)

    dicc=dict(zip(increase_c,suicide_rate_C))
    hh=sorted(dicc.items(),key=lambda x:x[0])
    increase_c_so=[]
    suicide_rate_C_so=[]
    for im in hh:
        increase_c_so.append(im[0])
        suicide_rate_C_so.append(item[1])

    from scipy.stats import kendalltau
    coef1, p1 = kendalltau(increase_c_so, suicide_rate_C_so)
    print('Kendall correlation coefficient: %.3f' % coef1)
    print('Samples are correlated (reject H0) p=%.3f' % p1)

    from scipy.stats import pearsonr
    coef3, p3 = pearsonr(increase_c_so, suicide_rate_C_so)
    print('Pearsonr correlation coefficient: %.3f' % coef3)
    print('Samples are correlated (reject H0) p=%.3f' % p3)

    return  coef3

cof_lit=[]
for country in country_list:
    cof_lit.append(Compute_country(country))
print(cof_lit)

