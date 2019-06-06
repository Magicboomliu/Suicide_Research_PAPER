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

# 都有哪些国家,生成词云
# country = list(data.Country)
# str=''
# for i in country:
#     str=str+i+" "
# print(country)
# from wordcloud import WordCloud
# wordcloud = WordCloud(background_color='white',width=1000,height=880).generate(str)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

#世界自杀率变化图
year_list=data.Year.unique()
year_list=sorted(year_list)
suicide_rate=[]
for year in year_list:
    pop_sum=data[data.Year==year].Population.sum()
    suicide_no=data[data.Year==year].SuicidesNo.sum()
    suicide_rate_year=round(suicide_no/pop_sum,7)
    suicide_rate.append(suicide_rate_year)
print(suicide_rate)
print(year_list)
year2=[]
for year in year_list:
    if year%2==0:
        year2.append("")
    else:
        year2.append(year)
print(year2)
plt.style.use('ggplot')
plt.xticks([i for i in range(len(year_list))],year2)
plt.title("Global Suicide Rate 1985-2015")
plt.xlabel("Year")
plt.ylabel("Suicide Rate of the Year")
plt.plot(suicide_rate,'b--',linewidth=3)
plt.show()


#全球整体不同年龄阶段的自杀率
l=[]
p15 = (data[(data["Age"] == '5-14 years')].Population.sum())
p155 = (data[(data["Age"] == '5-14 years')].SuicidesNo.sum())
p1 =  p155/p15
l.append(p1)
p25 = (data[ (data["Age"] == '15-24 years')].Population.sum())
p255 = (data[(data["Age"] == '15-24 years')].SuicidesNo.sum())
p2 =  p255/p25
l.append(p2)
p35 = (data[(data["Age"] == '25-34 years')].Population.sum())
p355 = (data[(data["Age"] == '25-34 years')].SuicidesNo.sum())
p3 =  p355/p35
l.append(p3)
p55 = (data[(data["Age"] == '35-54 years')].Population.sum())
p555 = (data[(data["Age"] == '35-54 years')].SuicidesNo.sum())
p5 =  p555/p55
l.append(p5)
p75 = (data[ (data["Age"] == '55-74 years')].Population.sum())
p755 = (data[(data["Age"] == '55-74 years')].SuicidesNo.sum())
p7 = p755/p75
l.append(p7)
p75plus = (data[(data["Age"] == '75+ years')].Population.sum())
p75pluss = (data[(data["Age"] == '75+ years')].SuicidesNo.sum())
p75P = p75pluss/p75plus
l.append(p75P)
plt.figure(figsize=(10,5))
plt.xlabel("Suicide rate")
plt.ylabel("Age")
plt.title("Age and Suicide Rate")
Age_list=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']
print([round(i ,6)for i in l])
sns.barplot(x=l,y=Age_list,alpha=0.6)
plt.show()
#
# sns.pairplot(data, hue="Age",kind='reg')
# plt.show()

male=data[data.Gender=='male'].Population.sum()
male_s=data[data.Gender=='male'].SuicidesNo.sum()
male_rate=male_s/male


female=data[data.Gender=='female'].Population.sum()
female_s=data[data.Gender=='female'].SuicidesNo.sum()
female_rate=female_s/female
sex_list=["male","female"]
rate_list=[]
rate_list.append(male_rate)
rate_list.append(female_rate)
sns.barplot(x=sex_list,y=rate_list,alpha=0.8)
plt.xlabel("Sex")
plt.ylabel("Suicide rate")
plt.title("Suicide Rate by Gender 1985-2015")
plt.show()
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
del increase[-1]
del year_list[-1]
plt.xlabel("Year")
plt.ylabel("Population increase Rate")
plt.title("Population increase Rate by year 1985-2015")
plt.plot(year_list,increase,'b--',linewidth=2)
plt.plot(year_list,[i*0 for i in range(len(year_list)) ],'r',linewidth=3)
print(increase)
plt.show()
