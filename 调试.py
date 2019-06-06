__author__ = "Luke Liu"
#encoding="utf-8"
_author__ = "Luke Liu"
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

sum=list(data[data.Country=="Albania"].GdpForYearMoney)

print(sum)
print(type(sum))