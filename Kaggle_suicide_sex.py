__author__ = "Luke Liu"
#encoding="utf-8"
import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
import seaborn as sns
import warnings
import  math
warnings.filterwarnings('ignore')
csv_file_path= 'master.csv'
data=pd.read_csv(csv_file_path)
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
data_withHDI=data[data['HDIForYear']>0]
# 删除HDI,因为大部分不包含
data=data.drop(['HDIForYear'],axis=1)
data=data.drop(['CountryYear'],axis=1)
#print(data.isnull().sum())
# First let's check different sex-composition whether affect the suicide_rate global
year_list=data.Year.unique()
year_list=sorted(year_list)
female_pop_global=[]
male_pop_global=[]
gender_rate=[]
for year in year_list:
    if year>=1987&year<=2010:
        female_pop=data[(data.Year==year)&(data.Gender=='female')].Population.sum()
        male_pop=data[(data.Year==year)&(data.Gender=='male')].Population.sum()
        female_pop_global.append(female_pop)
        male_pop_global.append(male_pop)
        rate=math.fabs(1-male_pop/female_pop)
        gender_rate.append(rate)
# plt.plot(year_list,female_pop_global,'r',label="female")
# plt.plot(year_list,male_pop_global,'b',label="male")
# plt.legend()
# plt.scatter(year_list,gender_rate,c='r',s=20)
# plt.show()
print(len(gender_rate))
suicide_year_globally=[]
for year in year_list:
    if year >= 1987 & year <= 2010:
        pop=suicide_number=data[data.Year==year].Population.sum()
        suicide_number=data[data.Year==year].SuicidesNo.sum()
        global_suicide_rate=suicide_number/pop
        suicide_year_globally.append(global_suicide_rate)
print(len(suicide_year_globally))


combine_x = dict(zip(gender_rate,suicide_year_globally))
h =sorted(combine_x.items(),key=lambda x:x[0])
gender_rate_so=[]
suicide_year_globally_so=[]
for i in h:
    gender_rate_so.append(i[0])
    suicide_year_globally_so.append(i[1])
print(gender_rate_so)
print(suicide_year_globally_so)
r=np.random.rand(len(gender_rate_so))
 #多线性回归
from  sklearn import model_selection
from sklearn import  metrics
from sklearn import linear_model
linereg01 = linear_model.LinearRegression()  # 生成一个线性回归实例
p1=np.array(gender_rate_so).reshape(-1,1)
p2=np.array(suicide_year_globally_so).reshape(-1,1)
linereg01.fit(p1,p2)
y_predict = linereg01.predict(p1)
plt.ylim(0.000075,0.000175)
plt.scatter(gender_rate_so,suicide_year_globally_so,s=100,c=r)
plt.xlabel("Gender_rate")
plt.ylabel("suicide_Rate")
plt.title("The relationship between Gender rate and suicide Rate")
plt.plot(p1,y_predict,label='predict_line')
R_value = linereg01.score(p1,p2)
print("The R^2 is {}".format(R_value))
plt.legend()
plt.show()

#Kendall秩相关
from scipy.stats import kendalltau
coef, p = kendalltau(gender_rate_so, suicide_year_globally_so)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)
#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(gender_rate_so, suicide_year_globally_so)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)
# Let's do it country by country in different country
# using albania as an example


from scipy.stats import pearsonr

coef3, p3 = pearsonr(gender_rate_so, suicide_year_globally_so)
print('Pearsonr correlation coefficient: %.3f' % coef3)
print('Samples are correlated (reject H0) p=%.3f' % p3)

print("_____________________________________________________")

def Compute_the_R_of_sex(countryname):

    year_list = data[data.Country==countryname].Year.unique()
    year_list=sorted(year_list)
    female_pops = []
    male_pops = []
    gender_Rate = []
    for year in year_list:
        male_pop=data[(data["Country"]==countryname)&(data["Gender"]=="male")&(data["Year"]==year)].Population.sum()
        female_pop=data[(data["Country"]==countryname)&(data["Gender"]=="female")&(data["Year"]==year)].Population.sum()
        female_pops.append(female_pop)
        male_pops.append(male_pop)
        gender_Rate.append(math.fabs(1-male_pop/female_pop))
    suicide_Rate=[]
    for year in year_list:
        pop=data[(data["Country"]==countryname)&(data["Year"]==year)].Population.sum()
        suicide_NO=data[(data["Country"]==countryname)&(data["Year"]==year)].SuicidesNo.sum()
        suicide_rates=suicide_NO/pop
        suicide_Rate.append(suicide_rates)
    combines=dict(zip(gender_Rate,suicide_Rate))
    h=sorted(combines.items(),key=lambda x:x[0])
    gender_Rate_so=[]
    suicide_Nos_so=[]
    for items in h:
        gender_Rate_so.append(items[0])
        suicide_Nos_so.append(items[1])
    # Kendall秩相关

    coef1, p = kendalltau(gender_Rate_so, suicide_Nos_so)
    print('Kendall correlation coefficient: %.3f' % coef1)
    print('Samples are correlated (reject H0) p=%.3f' % p)
    # Spearman相关分析,Spearman秩相关

    coef2, p = spearmanr(gender_Rate_so, suicide_Nos_so)
    print('Spearmans correlation coefficient: %.3f' % coef2)
    print('Samples are correlated (reject H0) p=%.3f' % p)
    return coef2


country_list=data.Country.unique()
R_list=[]
for country in country_list:
    R_list.append(Compute_the_R_of_sex(country))
combinessss=dict(zip(country_list,R_list))
Per_GDP_of_each_country=[]
for country in country_list:
    min_year = data[(data["Country"]==country)].Year.min()
    max_year =data[data["Country"]==country].Year.max()
    ggaps=max_year-min_year+1
    Per_GDP=0
    for year in year_list:
        vales=data[(data["Country"] == country) & (data["Year"] == year) & (data["Age"] == "15-24 years")].GdpPerCapitalMoney.sum()
        Per_GDP+=vales/2
    Per_GDP=Per_GDP/ggaps

    Per_GDP_of_each_country.append(Per_GDP)
R_list_dict=dict(zip(country_list,R_list))
per_gdp_dict=dict(zip(country_list,Per_GDP_of_each_country))
temp1=sorted(per_gdp_dict.items(),key=lambda  x:x[1])
C_names=[]
C_vales_b=[]
for items in temp1:
    C_names.append(items[0])
    C_vales_b.append(items[1])
R_value_so_list=[]
for name in C_names:
    R_value_so_list.append(R_list_dict[name])
print(len(R_value_so_list))
print(R_value_so_list)
print(C_vales_b)

for i in range(len(R_value_so_list)):
    if R_value_so_list[i]=='':
        del R_value_so_list[i]
        del C_vales_b[i]

data2=[-0.4727272727272727, 0.5186324786324785, -0.23207227555053642, -0.3871337719047421, 0.5882352941176471, 0.11428571428571425, -0.08189777225761523, -0.7676923076923077, 0.07013584757540453, -0.2100602137509042, -0.7772304324028462, 0.8406680016960503, -0.29365265886412645, 0.947641592478707, 0.43403781979977757, -0.2974559757206353, -0.9999999999999999, -0.07339901477832511, -0.4311688311688312, 0.2851669403393541, -0.4583123322011038, 0.004555820475782052, -0.7775305895439377, 0.7346774193548389, -0.3921529585108734, -0.36536356938495795, -0.7562980323611617, 'nan', 0.4566116604886569, 0.6814525057810461, 0.17647058823529413, -0.18174353529988538, 0.37226904914313447, -0.8064085646908278, -0.2246153846153846, -0.4351674042778922, 'nan', -0.5227920702513205, 0.08380650921724955, 0.8504032258064517, -0.3269165302212949, -0.8687179487179486, 0.7969348659003832, 0.07822580645161291, -0.753225806451613, 0.49999999999999994, 0.7248714625002061, 0.133455503389982, -0.8831168831168831, 0.3071796613699292, -0.9590511537477974, -0.7107692307692307, -0.026553676554259425, -0.8660254037844387, 0.5457627989662681, -0.11667115548630112, 0.6560571960173381, -0.6071428571428572, 0.7415661817431636, 0.23927198384073775, 0.5671550671550671, -0.7561789436999957, -0.590725806451613, 0.514014932814372, 'nan', -0.4987892233175538, 0.9194805194805193, -0.022451794424731678, -0.3931451612903226, 0.25080069743641176, 0.0, 0.8146851227266276, 0.43451961103976183, 0.2928714492293262, -0.21895161290322585, -0.5389162561576353, -0.8576195773081201, 0.46073414905450494, 0.14475806451612905, 0.298164266829872, 0.5237512547705973, 0.9593108504398827, 0.9165811965811965, 0.9655172413793103, -0.10405323865524362, 0.5423387096774195, 0.3823737192203587, -0.8233870967741936, 0.31695090616231114, 0.7782705335968808, 0.8285714285714287, 0.8107142857142855, 0.8782830059351149, 0.22321249611010346, 0.8025974025974025, 0.38688213440756564]
data1=[452.1363636363636, 720.7307692307693, 859.04, 875.9090909090909, 893.4444444444445, 1155.15, 1601.8846153846155, 1610.5, 1704.125, 1746.9259259259259, 1803.1379310344828, 2028.8333333333333, 2078.5555555555557, 2129.2, 2263.5333333333333, 2448.64, 2538.75, 2618.103448275862, 2800.48, 3142.0, 3286.2580645161293, 3587.1666666666665, 3640.4333333333334, 3708.967741935484, 3739.366666666667, 3995.6774193548385, 4061.8333333333335, 4124.0, 4195.6, 4351.166666666667, 4471.277777777777, 4822.571428571428, 5069.25, 5149.7, 5329.115384615385, 5343.967741935484, 5525.0, 5533.03125, 5589.413793103448, 6091.4838709677415, 6187.193548387097, 6518.814814814815, 6884.451612903225, 7138.451612903225, 7493.064516129032, 7519.807692307692, 7914.096774193548, 8829.037037037036, 8961.095238095239, 9100.032258064517, 9329.5, 9396.192307692309, 10068.347826086956, 10370.333333333334, 10375.181818181818, 10561.206896551725, 11376.095238095239, 12134.285714285714, 12413.592592592593, 12627.666666666666, 12758.666666666666, 14044.41935483871, 14801.258064516129, 17019.387096774193, 18081.0, 18352.645161290322, 18642.238095238095, 19947.235294117647, 20982.16129032258, 22279.48275862069, 22998.714285714286, 23033.055555555555, 23125.451612903227, 23182.5, 26602.58064516129, 30887.48275862069, 31481.466666666667, 31719.09677419355, 31908.354838709678, 32066.74193548387, 34230.86666666667, 34328.0, 35164.230769230766, 35468.275862068964, 35779.21875, 36397.54838709677, 38050.25806451613, 39269.6129032258, 39407.21875, 41436.666666666664, 42162.0, 46157.63636363636, 49299.90909090909, 57319.6, 62981.76190476191, 68798.3870967742]
#Kendall秩相关
print("_____________________________________________")
from scipy.stats import kendalltau
coef, p = kendalltau(data1, data2)
print('Kendall correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)
#皮尔森相关分析,Spearman秩相关
from scipy.stats import spearmanr
coef, p = spearmanr(data1, data2)
print('Spearmans correlation coefficient: %.3f' % coef)
print('Samples are correlated (reject H0) p=%.3f' % p)

