__author__ = "Luke Liu"
#encoding="utf-8"
import  os
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

# ok ,开始进行数据分析
#（1）确定数据中的最大年份与最小年份
max_year=max(data.Year)
min_year=min(data.Year)

# First analyse the 1975 data
data_country=data[(data['Year']==min_year)]
# 1985年的国家列表（唯一值）
country_1985=data[(data['Year']==min_year)].Country.unique()
# 1985 年男性列表 与 1985 年的女性列表
country_1985_male=[]
country_1985_female=[]
# 统计出不同国家男性与女性的年龄分段。
for country in country_1985:
    country_1985_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_1985_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')]))
# plt.figure(figsize=(10,10))
# sns.barplot(y=country_1985,x=country_1985_male,color='black')
# sns.barplot(y=country_1985,x=country_1985_female,color='red')
# plt.ylabel('Countries')
# plt.xlabel('Count Male vs Female')
# plt.title('1985 Year Suicide Rate Gender')
# plt.show()
#

# Very odd all the rates came on an equal level. So let's do max year.

data_country = data[(data['Year'] == max_year)]

country_2016 = data[(data['Year'] == max_year)].Country.unique()
country_2016_male = []
country_2016_female = []

for country in country_2016:
    country_2016_male.append(
        len(data_country[(data_country['Country'] == country) & (data_country['Gender'] == 'male')]))
    country_2016_female.append(
        len(data_country[(data_country['Country'] == country) & (data_country['Gender'] == 'female')]))

# We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.
# plt.figure(figsize=(10, 10))
# sns.barplot(y=country_2016, x=country_2016_male, color='red')
# sns.barplot(y=country_2016, x=country_2016_female, color='yellow')
# plt.ylabel('Countries')
# plt.xlabel('Count Male vs Female')
# plt.title('2016 Year Suicide Rate Gender')
# plt.show()

data_country=data[(data['Year']==min_year)]

country_1985_population=[]

for country in country_1985:
    country_1985_population.append(sum(data_country[(data_country['Country']==country)].Population))

#Now year 1985 find sum population every country

plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_population)
plt.xlabel('Population Count')
plt.ylabel('Countries')
plt.title('1985 Year Sum Population for Suicide Rate')
plt.show()

#每个国家人口的增长
country_lst=data.Country.unique()
data_1985_male_sum= data[(data.Country=='Albania')&(data.Year==1987)&(data.Gender=='male')].Population.sum()
data_1985_female_sum =data[(data.Country=='Albania')&(data.Year==1987)&(data.Gender=='female')].Population.sum()
pop_increase_list=[]
for country_name in country_lst:
    min_year=min(data[data.Country==country_name].Year)
    max_year=max(data[data.Country==country_name].Year)
    year_gaps=max_year-min_year
    data_poplution_previous=data[(data.Country==country_name)&(data.Year==min_year)].Population.sum()
    data_poplution_now = data[(data.Country == country_name) & (data.Year == max_year)].Population.sum()
    data_pop_increas=data_poplution_now-data_poplution_previous
    if data_poplution_previous==0:
        data_poplution_previous=1
    data_pop_increas_rate=round(data_pop_increas/data_poplution_previous,4)
    pop_increase_list.append(data_pop_increas_rate)
# 每个国家自杀率的增长
data_suicide_no_sum_rate=[]
for country_name in  country_lst:
    min_year = min(data[data.Country == country_name].Year)
    max_year = max(data[data.Country == country_name].Year)
    year_gap = max_year - min_year
    data_poplution_previous = data[(data.Country == country_name) & (data.Year == min_year)].Population.sum()
    data_poplution_now = data[(data.Country == country_name) & (data.Year == max_year)].Population.sum()
    data_suicide_no_prveious=data[(data.Country==country_name)&(data.Year==min_year)].SuicidesNo.sum()
    data_suicide_no_previous_rate=data_suicide_no_prveious/data_poplution_previous
    data_suicide_no_now=data[(data.Country==country_name)&(data.Year==max_year)].SuicidesNo.sum()
    data_suicide_no_now_rate=data_suicide_no_now/data_poplution_now
    if data_suicide_no_now_rate==0:
        data_suicide_no_now_rate=0.00001
    if data_suicide_no_previous_rate==0:
        data_suicide_no_previous_rate=0.00001

    suicide_increase_rate=round((data_suicide_no_now_rate-data_suicide_no_previous_rate)/data_suicide_no_now_rate,4)
    data_suicide_no_sum_rate.append(suicide_increase_rate)
# print((pop_increase_list))
# print((data_suicide_no_sum_rate))
suicide_rate_and_poplution_increase=dict(zip(pop_increase_list,data_suicide_no_sum_rate))
h=sorted(suicide_rate_and_poplution_increase.items(),key=lambda x:x[0])
pop_increase_list_so=[]
data_suicide_no_sum_rate_so=[]
for items in h:
    if(items[0]<=2):
        pop_increase_list_so.append(items[0])
        data_suicide_no_sum_rate_so.append(items[1])

#利用多项式拟合得出的结果
plt.ylim(-3,2)
plt.xlim(-0.5,1.8)
plt.scatter(pop_increase_list_so,data_suicide_no_sum_rate_so,c='r',s=60,label='orginal_data') #画出散点图
#系数为4的拟合
p1=np.polyfit(pop_increase_list_so,data_suicide_no_sum_rate_so,6)   #计算系数为4的拟合参数
z1=np.polyval(p1,pop_increase_list_so)          #生成拟合度为4的预测值的集合
plt.plot(pop_increase_list_so,z1,'b',label='parameters=6')  #画出曲线
plt.legend()
plt.show()
from scipy.stats import kendalltau
coef, p = kendalltau(pop_increase_list_so, data_suicide_no_sum_rate_so)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)
#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(pop_increase_list_so, data_suicide_no_sum_rate_so)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

# #多线性回归
# from  sklearn import model_selection
# from sklearn import  metrics
#
# from sklearn import linear_model
#
# linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
# p1=np.array(pop_increase_list_so).reshape(-1,1)
# p2=np.array(data_suicide_no_sum_rate_so).reshape(-1,1)
#
# linereg01.fit(p1,p2)
# y_predict = linereg01.predict(p1)
# plt.ylim(-3,2)
# plt.xlim(-0.5,1.8)
# plt.scatter(pop_increase_list_so,data_suicide_no_sum_rate_so,c='r',s=60,label='orginal_data') #画出散点图
# plt.plot(p1,y_predict,label='predict_line')
# R_value = linereg01.score(p1,p2)
# print("The R^2 is {}".format(R_value))
# plt.legend()
# plt.show()




