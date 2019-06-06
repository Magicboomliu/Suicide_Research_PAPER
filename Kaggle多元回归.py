__author__ = "Luke Liu"
#encoding="utf-8"
__author__ = "Luke Liu"
#encoding="utf-8"
import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
import seaborn as sns
import warnings
import  math




def Guiyi(ls):
    a=np.array(ls)
    b=a.mean()
    c=a.std()
    d=(a-b)/c
    return d

warnings.filterwarnings('ignore')
csv_file_path= 'master.csv'
data=pd.read_csv(csv_file_path)
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
data_withHDI=data[data['HDIForYear']>0]
# 删除HDI,因为大部分不包含
data=data.drop(['HDIForYear'],axis=1)
data=data.drop(['CountryYear'],axis=1)

year_list=data.Year.unique()
year_list=sorted(year_list)
country_list=data.Country.unique()
population_list=[]
Age_list=[]
Sex_comp=[]

per_GDP=[]
suicide_rate=[]
for country in country_list:
    pop=data[data.Country==country].Population.sum()
    p15=(data[(data.Country==country)&(data.Age=='5-14 years')].Population.sum())/pop
    p25 = (data[(data.Country == country) & (data.Age == '15-24 years')].Population.sum()) / pop
    p35 = (data[(data.Country == country) & (data.Age == '25-34 years')].Population.sum()) / pop
    p55 = (data[(data.Country == country) & (data.Age == '35-54 years')].Population.sum()) / pop
    p75 = (data[(data.Country == country) & (data.Age == '55-74 years')].Population.sum()) / pop
    p75plus = (data[(data.Country == country) & (data.Age == '75+ years')].Population.sum()) / pop
    ages=p15*1+p25*2+p35*3+p55*4+p75*5+p75plus*6
    male=(data[(data.Country==country)&(data.Gender=='male')].Population.sum())/pop
    female=(data[(data.Country==country)&(data.Gender=='female')].Population.sum())/pop
    sex_rate=male/female

    per_GDPs=data[data.Country==country].GdpPerCapitalMoney.sum()
    suicide=data[data.Country==country].SuicidesNo.sum()
    suicide_rates=suicide/pop

    #append操作
    population_list.append(pop)
    Age_list.append(ages)
    Sex_comp.append(sex_rate)

    per_GDP.append(per_GDPs)
    suicide_rate.append(suicide_rates)



fe1=Guiyi(population_list)
fe2=Guiyi(Age_list)
fe3=Guiyi(Sex_comp)
fe4=Guiyi(per_GDP)
target=Guiyi(suicide_rate)

datas=np.column_stack((fe1,fe2,fe3,fe4))
print(datas.shape)
print(target.shape)


#多元线性分析

from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model

linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例

# 分割模型为训练集与测试集（9:1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    datas, target, test_size=0.1, random_state=1000
)

# 训练测试集（利用 gradient desecnt 寻找 w 与 b)

linereg01.fit(X_train, y_train)
y_predict_in_train = linereg01.predict(X_train)
y_predict_in_test = linereg01.predict(X_test)
w = linereg01.coef_  # 得到权重列表
b = linereg01.intercept_  # 得到bias值
print(len(w))  # 输出参数数目
print([round(i, 5) for i in w])  # 输出w列表，保留5位小数
print(b)  # 输出bias
error_in_train = metrics.mean_squared_error(y_predict_in_train, y_train)  # 训练集上的Loss fucntion值（mean square）
error_in_test = metrics.mean_squared_error(y_predict_in_test, y_test)  # 测试集上 Loss function的值（mean square)
R_value = linereg01.score(X_train, y_train)  # 计算 X与y 的R 相关指数的大小
print("error in train:{}".format(error_in_train))
print("error in test:{}".format(error_in_test))
# 我们将二者的拟合程度画出来
plt.plot(y_test, linewidth=3, label='Raw data')
plt.plot(y_predict_in_test, linewidth=3, label="Multiple linear regression")
plt.legend()
plt.title("Performance on the test set")
plt.ylabel("Suicide Rate")
plt.show()

#在原始的数据集上进行拟合

plt.plot(y_train, linewidth=3, label='Raw data')
plt.plot(y_predict_in_train, linewidth=3, label="Multiple linear regression")
plt.legend()
plt.title("Performance on the train set")
plt.ylabel("Suicide Rate")
plt.show()

#随机森林
from sklearn import tree
dtc=tree.DecisionTreeRegressor()
dtc.fit(X_train,y_train)
sc = dtc.score(X_train,y_train)
with open("tree1.dot",'w') as f:
    f=tree.export_graphviz(dtc,out_file=f)
y_predict_in_train_tree=dtc.predict(X_train)
y_predict_in_test_tree=dtc.predict(X_test)




plt.plot(y_train, linewidth=3, label='Raw data')
plt.plot(y_predict_in_train_tree, linewidth=1, label=" Decision Tree")
plt.legend()
plt.title("Performance on the train set")
plt.ylabel("Suicide Rate")
plt.show()



plt.plot(y_test, linewidth=3, label='Raw data')
plt.plot(y_predict_in_test_tree, linewidth=1, label="Decision Tree")
plt.legend()
plt.title("Performance on the test set")
plt.ylabel("Suicide Rate")
plt.show()
error_in_train_s = metrics.mean_squared_error(y_predict_in_train_tree, y_train)  # 训练集上的Loss fucntion值（mean square）
error_in_test_s = metrics.mean_squared_error(y_predict_in_test_tree, y_test)  # 测试集上 Loss function的值（mean square)
print(error_in_test_s)
print(error_in_train_s)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(X_train, y_train)

rfr_y_predict_s = rfr.predict(X_test)
rfr_y_predict_s2 = rfr.predict(X_train)

plt.plot(y_train, linewidth=3, label='Raw data')
plt.plot(rfr_y_predict_s2, linewidth=1, label=" Random Forest")
plt.legend()
plt.title("Performance on the train set")
plt.ylabel("Suicide Rate")
plt.show()



plt.plot(y_test, linewidth=3, label='Raw data')
plt.plot(rfr_y_predict_s , linewidth=1, label="Random Forest")
plt.legend()
plt.title("Performance on the test set")
plt.ylabel("Suicide Rate")
plt.show()
error_in_train_1 = metrics.mean_squared_error(rfr_y_predict_s2, y_train)  # 训练集上的Loss fucntion值（mean square）
error_in_test_1 = metrics.mean_squared_error(rfr_y_predict_s, y_test)  # 测试集上 Loss function的值（mean square)
print(error_in_test_1)
print(error_in_train_1)
with open("RF.dot",'w') as f1:
    f1=tree.export_graphviz(rfr,out_file=f1)