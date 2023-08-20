#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer,KNNImputer
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import xgboost as xgb


# In[2]:


import xgboost as xgb


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[4]:


df = pd.read_csv('Mobile Price Prediction.csv')


# In[5]:


df.sample(5)


# In[6]:


df.info()


# In[7]:


df1 = df.copy()


# In[8]:


df1.drop(['model','2G_bands','3G_bands','4G_bands','dimentions','weight_oz','display_size','OS','Chipset','GPU','loud_speaker','bluetooth','GPS','radio','battery','colors','img_url'],axis=1,inplace=True)


# In[9]:


df1.info()


# In[10]:


label = LabelEncoder()


# In[11]:


def encode(x):
    return label.fit_transform(x)


# In[12]:


def unique(x):
    return x.unique()


# In[13]:


len(list(unique(df1['brand'])))


# In[14]:


df1.NFC.unique()


# In[15]:


df1['NFC'].replace({np.nan:'No','TBD':'No','To be confirmed':'No','Region specific':'Yes','Carrier dependent':'Yes','TBC':'No','Optional':'Yes','Via software update':'No','O2 UK only':'No','NFC':'Yes'},inplace=True)


# In[16]:


df1.NFC.value_counts()


# In[17]:


nfc = df1['NFC'].str.split(' ',expand=True)[0]


# In[18]:


nfc.replace('\/','',inplace=True,regex=True)


# In[19]:


nfc.unique()


# In[20]:


df1['NFC'] = encode(nfc)


# In[21]:


df1['NFC'].unique()


# In[22]:


df1['sensors'].unique()


# In[23]:


df1['internal_memory'].unique()


# In[24]:


internal = df1['internal_memory'].str.split(' ',expand=True)[0]


# In[25]:


internal.unique()


# In[26]:


internal.replace({'ROM':158,'16/32':16000,'32':32000,'8':8000,'16':16000,'11':11000,'16/32/64':3200,'32/64':32000,'8/16':8000,'8/16/32':8000,'4/16':4000,'64/256/512':256000,'32/128':128000,'32/128/256':128000,'16/32/64/128':64000,'16/64/128':64000,'8/16/32/64':32000,'4/8/16':8000,'64/128/256':128000,'32/64/128':64000,'64/128':64000,'4/8':4000,'GB':000,'128/256':128000,'40/80':40,'16/64':16000,'2/4':2000,'Yes':10000,'1/2':512,'2/8':2000,'2/8/16':2000,'20/40':20},inplace=True)


# In[27]:


internal.replace({np.nan:100000,'No':1,'64/256':64000},inplace=True)


# In[28]:


internal.replace('GB',000,inplace=True,regex=True)


# In[29]:


internal.replace('',100000,inplace=True,regex=True)


# In[30]:


internal.replace({100000:32,10000:32},inplace=True)


# In[31]:


internal = internal.astype(float,6)


# In[32]:


internal.median()


# In[33]:


df1['internal_memory'] = internal


# In[34]:


df1.info()


# - primary camera

# In[35]:


secondary = df1['secondary_camera'].str.split(' ',expand=True)[[0,1]]


# In[36]:


secondary.sample(10)


# In[37]:


secondary[0].unique()


# In[38]:


secondary[1].unique()


# In[39]:


secondary[0].replace({'VGA/':0.3,'VGA':0.3,'QVGA':0.08,'Videocall':5,'CIF':0.2,'Videocalling':5,'8MP':8,np.nan:10000,'VGA@15fps':2,'Spy':1,'QCIF':0.5,'QCIF@15fps':1,'Dual':10000,'720p':13,'1.3MP':1.3,'HD':1.6,'No':0,'Yes':10000},inplace=True)


# In[40]:


secondary[0].unique()


# In[41]:


secondary = secondary[0].astype(float)


# In[42]:


secondary.median()


# In[43]:


secondary.mode()


# In[44]:


secondary.replace({10000:0.3},inplace=True)


# In[45]:


df1['secondary_camera'] = secondary


# In[46]:


df1.info()


# In[47]:


df1['network_technology'].unique()


# In[48]:


df2 = df1[df1['network_technology']!='No cellular connectivity']


# In[49]:


df2.reset_index(drop=True,inplace=True)


# In[50]:


df2.index


# In[51]:


df2['network_technology'].unique()


# In[52]:


network_technology = df2['network_technology'].str.split(' /',expand=True)


# In[53]:


network_technology.columns = ['network1','network2','network3','network4','network5']


# In[54]:


network_technology['network1'].unique()


# In[55]:


network_technology['network2'].unique()


# In[56]:


network_technology['network2'].replace({np.nan:'No',' HSPA':'HSPA',' UMTS':'UMTS',' EVDO':'EVDO',' CDMA':'CDMA',' LTE':'LTE',' CDMA2000':'CDMA'},inplace=True)


# In[57]:


network_technology['network2'].unique()


# In[58]:


network_technology['network3'].unique()


# In[59]:


network_technology['network3'].replace({np.nan:'No',' LTE':'LTE',' HSPA':'HSPA',' EVDO':'EVDO',' UMTS':'UMTS',' CMDA2000':'CMDA2000'},inplace=True)


# In[60]:


network_technology['network3'].unique()


# In[61]:


network_technology['network4'].unique()


# In[62]:


network_technology['network4'].replace({np.nan:'No',' LTE':'LTE',' EVDO':'EVDO',' CDMA2000':'CDMA'},inplace=True)


# In[63]:


network_technology['network4'].unique()


# In[64]:


network_technology['network5'].unique()


# In[65]:


network_technology['network5'].replace({np.nan:'No'},inplace=True)


# In[66]:


network_technology['network5'].unique()


# In[67]:


network1 = pd.DataFrame(encode(network_technology['network1']),columns=['network1'])


# In[68]:


network2 = pd.DataFrame(encode(network_technology['network2']),columns=['network2'])


# In[69]:


network3 = pd.DataFrame(encode(network_technology['network3']),columns=['network3'])


# In[70]:


network4 = pd.DataFrame(encode(network_technology['network4']),columns=['network4'])


# In[71]:


network5 = pd.DataFrame(encode(network_technology['network5']),columns=['network5'])


# In[72]:


df2.drop('network_technology',axis=1,inplace=True)


# In[73]:


df2.insert(loc=1,value=network1,column='network1')


# In[74]:


df2.insert(loc=2,value=network2,column='network2')


# In[75]:


df2.insert(loc=3,value=network3,column='network3')


# In[76]:


df2.insert(loc=4,value=network4,column='network4')


# In[77]:


df2.insert(loc=5,value=network5,column='network5')


# In[78]:


df2.info()


# In[79]:


df2['sensors'].unique()


# In[80]:


sensors = df2['sensors'].str.split('\/',expand=True)


# In[81]:


sensors[0].nunique()


# In[82]:


sensors[1].nunique()


# In[83]:


sensors[2].nunique()


# In[84]:


sensors[3].nunique()


# In[85]:


sensors[4].nunique()


# In[86]:


sensors[5].nunique()


# In[87]:


sensors[6].nunique()


# In[88]:


sensors[7].nunique()


# In[89]:


sensors[8].nunique()


# In[90]:


sensors[9].nunique()


# In[91]:


announced = (df2['announced'].str.split(' ',expand = True))


# In[92]:


year = pd.DataFrame(announced[0])


# In[93]:


year.value_counts()


# In[94]:


year.replace({'Exp.':2014,'Not':0,'2Q':2003,'3Q':2002,'':1,'2007.':2007,'2008.':2008,'2009.':2009,'2010.':2010,'2011.':2011,'1Q':2004,'4Q':2002,'Q3':2001,'Feb-01':0,'Q2':2001,'Jun-02':0,'Sep-02':0,'Oct-02':0,'Feb-02':0,'Nov-02':0,'Apr-02':0,'Apr-04':0,'Jun-03':0,'May-01':0,'May-02':0,'Mid':0,'Never':1,'Q4':2002,'2006.':2006},inplace=True)


# In[95]:


year.replace({1:2014,np.nan:2014},inplace=True)


# In[96]:


year = year.astype(int)


# In[97]:


year = year.astype(int)


# In[98]:


df2['announced'] = year


# In[99]:


df2['announced'].replace({0:2014},inplace=True)


# In[100]:


df2['announced'].value_counts()


# In[101]:


status = df2['status'].str.split(' ',expand=True)


# In[102]:


status[0].unique()


# In[103]:


status[0].replace({'Available.':'Available'},inplace=True)


# In[104]:


status[0].value_counts()


# In[105]:


status = status[0]


# In[106]:


df2['status'] = status


# In[107]:


df2['status'] = encode(df2['status'])


# In[108]:


df2.info()


# In[109]:


df2['brand'] = encode(df2['brand'])


# In[110]:


memory_card = df2['memory_card'].str.split(' ',expand=True)


# In[111]:


memory_card[0].unique()


# In[112]:


memory_card.replace({'3-in-1':1,'No':0,'Adreno':10,'To':1,'Yes':1,'Memory':1,'2':'microSD'},inplace=True)


# In[113]:


memory_card[0].unique()


# In[114]:


memory_card[0].replace({1:'microSD','MMC-micro':'microMMC',0:'No',10:'not memory card'},inplace=True)


# In[115]:


df2['memory_card'] = encode(memory_card[0])


# In[116]:


df2.info()


# In[117]:


df2['USB'].unique()


# In[118]:


df2[df2['USB']=='miniUSB (charging only)']['USB'].head()


# In[119]:


df2['USB'].replace({'USB 2.0':'USB TYPE A','miniUSB':'USB TYPE B','miniUSB 2.0':'USB TYPE B','Type-C 1.0 reversible connector':'USB TYPE C',
            'Type-C 1.0 reversible connector/ USB On-The-Go':'USB TYPE C','Type-C 1.0 reversible connector/ magnetic connector':'USB TYPE C','Type-C 1.0 reversible connector/ USB On-The-Go':'USB TYPE C',
            'Type-C 1.0 reversible connector/ USB On-The-Go; magnetic connector':'USB TYPE C','Type-C 1.0 reversible connector (MHL2 TV-out)':'USB TYPE C',
            'Type-C 1.0 reversible connector; magnetic connector':'USB TYPE C','Type-C 1.0 reversible connector; USB Host':'USB TYPE C',
            'Type-C 1.0 reversible connector; USB On-The-Go':'USB TYPE C','Type-C 1.0 reversible connector/ USB On-The-Go; magnetic connector':'USB TYPE C',
            'Type-C 1.0 reversible connector; USB Host':'USB TYPE C','3.1/ Type-C 1.0 reversible connector':'USB TYPE C','3.1/ Type-C 1.0 reversible connector/ USB On-The-Go':'USB TYPE C',
            '3.1/ Type-C 1.0 reversible connector; USB Host':'USB TYPE C','miniUSB (charging only)':'MINIUSB','miniUSB 2.0 (charging only)':'MINIUSB',
            'miniUSB 1.1, miniUSB/ USB host':'MINIUSB','miniUSB 2.0/ USB Host':'MINIUSB','miniUSB 2.0/ USB 2.0':'MINIUSB','miniUSB 2.0/ USB 2.0':'MINIUSB',
            'microUSB 2.0':'MICROUSB','microUSB 2.0/ USB On-The-Go':'MICROUSB','microUSB 2.0/ USB Host':'MICROUSB','microUSB 1.1':'MICROUSB','microUSB':'MICROUSB','microUSB (charging only)':'MICROUSB',
            'microUSB (charging and headset only)':'MICROUSB','microUSB 3.0':'MICROUSB','microUSB/ USB Host':'MICROUSB','microUSB 2.0 (SlimPort)':'MICROUSB','microUSB 2.0 (SlimPort TV-out)/ USB Host':'MICROUSB',
            'microUSB 2.0 (SlimPort TV-out)':'MICROUSB','microUSB 2.0 (MHL TV-out)':'MICROUSB','microUSB v2.0/ USB On-The-Go':'MICROUSB','microUSB 3.0/ USB Host':'MICROUSB','microUSB 2.0 HS':'MICROUSB',
            'microUSB 2.0 (charging only)':'MICROUSB','microUSB 2.0 (SlimPort 4K)':'MICROUSB','microUSB 2.0 (SlimPort 4K)/ USB On-The-Go':'MICROUSB','microUSB 2.0 (SlimPort 4K)/ USB Host':'MICROUSB',
            'microUSB 2.0 (MHL 2 TV-out)/ USB Host':'MICROUSB','microUSB 2.0 (MHL TV-out)/ USB Host':'MICROUSB','microUSB 2.0 (MHL 3 TV-out)/ USB Host':'MICROUSB',
            'microUSB 2.0 (MHL 2.1 TV-out)/ USB Host':'MICROUSB','microUSB 2.0 (MHL 1.3 TV-out)/ USB Host':'MICROUSB','microUSB 3.0 (MHL TV-out)/ USB Host':'MICROUSB',
            'microUSB 2.0 (MHL)':'MICROUSB',' microUSB 3.0 (MHL 2 TV-out)/ USB Host':'MICROUSB',' microUSB 2.0 (MHL 1.2)/ USB Host':'MICROUSB',
            'Pop-Port':'OTHERS',' Pop-Port 2.0':'OTHERS','Pop-Port 1.1':'OTHERS','Yes/':10,'Yes (optional)':10,'Yes 2.0':10,'TBC':'NONE','1':10,'1.1 (USB host)':'MICROUSB','2.0 (TV-out)':'MICROUSB','2.0/ reversible connector':'USB TYPE C',
            '2.0/ reversible connector; magnetic connector':'USB TYPE C','2.0/ Type-C 1.0 reversible connector/ USB':'USB TYPE C','microUSB 2.0/ USB 2.0':'MICROUSB'},inplace=True)


# In[120]:


df2['USB'].unique()


# In[121]:


df2['USB'].replace({'miniUSB 1.1':'MINIUSB','2.0/ Type-C 1.0 reversible connector':'USB TYPE C','3.0/ Type-C 1.0 reversible connector':'USB TYPE C',
                   '3.0/ reversible connector; magnetic connector':'USB TYPE C','3.0/ Type-C 1.0 reversible connector/ USB On-The-Go':'USB TYPE C',
                   '2.0/ Type-C 1.0 reversible connector/ USB On-The-Go':'USB TYPE C','microUSB 2.0/ USB Host/ USB OTG':'MICROUSB','3.0/ Type-C 1.0 reversible connector; magnetic connector':'USB TYPE C',
                   'microUSB 2.0 (MHL 3.0 TV-out)/ USB Host':'MICROUSB','microUSB 2.0 (MHL 3.0 TV-out)':'MICROUSB','Yes/ microUSB 2.0':'MICROUSB','USB host':'MICROUSB',
                   'microUSB 2.0 (SlimPort)/ USB Host':'MICROUSB','3.1/ Type-C 1.0 reversible connector/ magnetic connector/ USB On-The-Go':'USB TYPE C',
                   'Yes (charging only)':'MICROUSB','2.0/ USB host':'MICROUSB','USB 3.0/ USB Host':'MICROUSB','Pop-Port 2.0':'OTHERS','miniUSB/ USB host':'MINIUSB','microUSB 2.0; USB 2.0':'MICROUSB',
                   '3.1/ Type-C 1.0 reversible connector; magnetic connector':'USB TYPE C','microUSB 3.0 (MHL 2.1 TV-out)/ USB Host':'MICROUSB',
                   'microUSB 2.0 (MHL)/ USB Host':'MICROUSB','microUSB 2.0 (MHL 2.1)/ USB Host':'MICROUSB','microUSB 3.0 (MHL 2 TV-out)/ USB Host':'MICROUSB','microUSB 2.0 (MHL 1.2)/ USB Host':'MICROUSB',
                   '2.0/ USB Host':'MICROUSB','microUSB 2.0 (TV-out)/ USB Host':'MICROUSB','2.0 (TV-out)/ USB Host':'MICROUSB','2.0/ Type-C 1.0 reversible connector; USB Host':'USB TYPE C',
                   'microUSB 2.0 (MHL 3 TV-out)/ USB Host; magnetic connector':'MICROUSB','microUSB 2.0 (MHL TV-out)/ USB Host; magnetic connector':'MICROUSB','microUSB 2.0 (MHL 3 TV-out); magnetic connector':'MICROUSB',
                   'Yes (charging/ mass storage)':'MINIUSB','miniUSB 1.1/ USB host':'MINIUSB','2.0/ USB On-The-Go':'MINIUSB','charging only':'MINIUSB','microUSB 2.0 USB On-The-Go':'MICROUSB','1.1':'MICROUSB','1.2':'MICROUSB','Yes':10,'No':'None','2':'MICROUSB'},inplace=True)


# In[122]:


df2['USB'].replace({np.nan:'MICROUSB','OTHERS':'MINIUSB',10:'MINIUSB','None':'MICROUSB'},inplace=True)


# In[123]:


df2['USB'] = encode(df2['USB'])


# In[124]:


df2['USB'].unique()


# In[125]:


df2.info()


# In[126]:


df2.drop(['network_speed'],axis=1,inplace=True)


# In[127]:


df2.info()


# In[128]:


df2['sensors'].isnull().sum()


# In[129]:


sensors[0].unique()


# In[130]:


sensors[0].replace({'Accelerometer':'Motion sensor','Fingerprint (front-mounted)':'Biometric sensors1','Fingerprint (rear-mounted)':'Biometric sensors2','Fingerprint (front-mounted'
       :'Biometric sensors1','Proximity':'Proximity sensor','microUSB 2.0 (SlimPort)':'Other','Compass':'Environmental sensors','Gyro':'Motion sensor','Iris scanner':'Biometric Isensors','Fingerprint (side-mounted)':'Biometric sensors3','Fingerprint (rear-mounted; region dependent)':'Biometric sensors2','Ambient light sensor':'Environmental sensors','Thermometer':'Environmental sensors','Sound level meter':'Environmental sensors','Fingerprint (side-mounted':'Biometric sensors3','Fingerprint (side-mounted; EU model only)':'Biometric sensors3','Barometer':'Environmental sensors','Fingerprint (Exclusive edition only; side-mounted)':'Biometric sensors4','Yes':'fill','No':'Nothing'},inplace=True)


# In[131]:


sensors[0].unique()


# In[132]:


sensors[0].value_counts()


# In[133]:


sensors[0].replace({'fill':sensors[0].mode()[0]},inplace=True)


# In[134]:


sensors[0].replace({'Motion sensor':0,'Biometric sensors1':1,'Biometric sensors2':2,'Biometric sensors3':3,'Biometric sensors4':4,'Environmental sensors':5,
                   'Proximity sensor':6,'Nothing':7,'Other':8,'Biometric Isensors':9},inplace=True)


# In[135]:


imputer = KNNImputer(n_neighbors = 3)


# In[136]:


sensors[0] = imputer.fit_transform(sensors[[0]])


# In[137]:


sensors[0].unique()


# In[138]:


df2['sensors'] = sensors[0]


# In[139]:


df2.drop(['GPRS','EDGE'],axis=1,inplace=True)


# In[140]:


df2.info()


# In[141]:


df2.head(2)


# ### weight_g

# In[142]:


df2['weight_g'].unique()


# In[143]:


df2['weight_g'].isnull().sum()


# In[144]:


df2['weight_g'].mode()


# In[145]:


df2['weight_g'].fillna(df['weight_g'].mode()[0],inplace=True)


# In[146]:


df2['weight_g'].isnull().sum()


# In[147]:


weight = df2['weight_g'].str.split('-',expand=True)


# In[148]:


weight.sample(3)


# In[149]:


weight.sort_values([0]).tail(10)


# In[150]:


weight[0].replace({'From':weight[0].mode()[0]},inplace = True)


# In[151]:


weight.sort_values([0]).tail(10)


# In[152]:


weight[1].unique()


# In[153]:


weight[weight[1]=='236']


# In[154]:


weight[0].replace({'225':((225+236)/2)},inplace=True)


# In[155]:


weight[weight[1]=='185']


# In[156]:


weight[0].replace({'165':((165+185)/2)},inplace=True)


# In[157]:


weight1= pd.DataFrame(weight[0])


# In[158]:


weight1.sample(5)


# In[159]:


weight1[weight1[0]=='67/78']


# In[160]:


weight2 = weight1[0].str.split('/',expand =True)


# In[161]:


weight2[0].unique()


# In[162]:


weight2[1].unique()


# In[163]:


weight2[weight2[1]=='78']


# In[164]:


weight2[0].replace({'67':((67+78)/2)},inplace=True)


# In[165]:


weight3 = pd.DataFrame(weight2[0])


# In[166]:


weight3[0]=weight3[0].astype('float')


# In[167]:


weight3[0].dtypes


# In[168]:


weight3[0].mode()


# In[169]:


weight3[0].mean()


# In[170]:


weight3[0].fillna(weight3[0].mode()[0],inplace=True)


# In[171]:


df2['weight_g']  = weight3


# In[172]:


df2.info()


# ### SIM

# In[173]:


df2['SIM'].unique()


# In[174]:


sim = df2['SIM'].str.split('or',expand=True)


# In[175]:


sim


# In[176]:


sim1 = sim[0].str.split('SIM',expand=True)


# In[177]:


sim1


# In[178]:


sim1[0].replace({'\-': ''},regex = True,inplace = True)


# In[179]:


sim1


# In[180]:


sim1[0].unique()


# In[181]:


sim2 = sim1[0].str.split(' ',expand = True)


# In[182]:


sim2


# In[183]:


sim2[0].unique()


# In[184]:


sim3 = sim2[0].str.split('/',expand = True)


# In[185]:


sim3


# In[186]:


sim3[0].unique()


# In[187]:


sim4= pd.DataFrame(sim3[0])


# In[188]:


sim4.isnull().sum()


# In[189]:


sim4[0] = encode(sim4[0])


# In[190]:


sim4


# In[191]:


sim5 = sim[1].str.split('SIM',expand=True)


# In[192]:


sim5


# In[193]:


sim5[0].unique()


# In[194]:


sim5.isnull().sum()


# In[195]:


df2['SIM']= sim4


# In[196]:


df2.info()


# ### display_type

# In[197]:


df2['display_type'].unique()


# In[198]:


df2['display_type'].nunique()


# In[199]:


display = df2['display_type'].str.split('touchscreen',expand= True)


# In[200]:


display[0].unique()


# In[201]:


display2 = display[0].str.split('capacitive',expand=True)


# In[202]:


display2[0].unique()


# In[203]:


display2[0].nunique()


# In[204]:


display3= display2[0].str.split('colors',expand=True)


# In[205]:


display3[0].unique()


# In[206]:


display3[0].nunique()


# In[207]:


display3.isnull().sum()


# In[208]:


display3[0].fillna(display3[0].mode()[0],inplace=True)


# In[209]:


display4 = pd.DataFrame(display[0])


# In[210]:


display4[0] = encode(display4[0])


# In[211]:


display4.dtypes


# In[212]:


df2.columns


# In[213]:


df2['display_type']= display4


# In[214]:


df2.info()


# ### display_resolution  

# In[215]:


list(df['display_resolution'].unique())


# In[216]:


df['display_resolution'].nunique()


# In[217]:


resolution = df['display_resolution'].str.split(' ',expand=True)


# In[218]:


resolution[0].unique()


# In[219]:


resolution[resolution[0]=='LED-backlit']


# In[220]:


resolution[0].mode()


# In[221]:


resolution[0].replace({'LED-backlit':resolution[0].mode()[0]},inplace=True)


# In[222]:


resolution[0].unique()


# In[223]:


resolution[0].isnull().sum()


# In[224]:


resolution[0].replace({'/':np.nan},inplace=True)


# In[225]:


resolution[0] = imputer.fit_transform(resolution[[0]])


# In[226]:


resolution[0].unique()


# In[227]:


resolution_new = pd.DataFrame(resolution[0])


# In[228]:


resolution_new


# In[229]:


df2['display_resolution'] = resolution_new


# In[230]:


df2.info()


# ### CPU

# In[231]:


df2['CPU'].unique()


# In[232]:


cpu = df2['CPU'].str.split('core',expand=True)


# In[233]:


cpu[0].unique()


# In[234]:


cpu[0].nunique()


# In[235]:


cpu1 = cpu[0].str.split('-',expand=True)


# In[236]:


cpu1[0].unique()


# In[237]:


cpu2 = cpu1[0].str.split('Hz',expand=True)


# In[238]:


cpu2[0].unique()


# In[239]:


cpu2[0].nunique()


# In[240]:


cpu_new = pd.DataFrame(cpu2[0])


# In[241]:


cpu_new[0].unique()


# In[242]:


cpu_new[0] = encode(cpu_new[[0]])


# In[243]:


cpu_new[0].nunique()


# In[244]:


df2['CPU'] = cpu_new


# In[245]:


df2.info()


# ### RAM

# In[246]:


df2['RAM'].unique()


# In[247]:


ram = df2['RAM'].str.split('RAM',expand=True)


# In[248]:


ram


# In[249]:


ram[0].unique()


# In[250]:


ram[0].isnull().sum()


# In[251]:


ram[1].isnull().sum()


# In[252]:


ram[1].unique()


# In[253]:


ram1 = ram[0].str.split('B',expand =True)


# In[254]:


ram1


# In[255]:


ram1[0].unique()


# In[256]:


ram1[0].nunique()


# In[257]:


ram1[0].mode()


# In[258]:


ram1[0].fillna(ram1[0].mode()[0],inplace=True)


# In[259]:


ram1[0].isnull().sum()


# In[260]:


ram_new = pd.DataFrame(ram1[0])


# In[261]:


ram_new


# In[262]:


ram_new[0] = encode(ram_new[0])


# In[263]:


df2['RAM'] = ram_new


# In[264]:


df2.info()


# ### Primary_camera

# In[265]:


df2['primary_camera'].unique()


# In[266]:


cam = df2['primary_camera'].str.split('MP',expand=True)


# In[267]:


cam


# In[268]:


cam[0].unique()


# In[269]:


cam[0].nunique()


# In[270]:


cam[0].isnull().sum()


# In[271]:


cam[0].mode()


# In[272]:


cam[0].fillna(cam[0].mode()[0],inplace=True)


# In[273]:


cam[0].isnull().sum()


# In[274]:


cam_new = pd.DataFrame(cam[0])


# In[275]:


cam_new[0] = encode(cam_new[0])


# In[276]:


df2['primary_camera'] = cam_new


# In[277]:


df2.info()


# ### audio_jack          

# In[278]:


df2['audio_jack'].unique()


# In[279]:


df2['audio_jack'].isnull().sum()


# In[280]:


df2['audio_jack'].fillna(df2['audio_jack'].mode()[0],inplace=True)


# In[281]:


df2['audio_jack'].isnull().sum()


# In[282]:


df2['audio_jack'] = encode(df2['audio_jack'])


# In[283]:


df2.info()


# ###  WLAN

# In[284]:


df2['WLAN'].unique()


# In[285]:


wifi = df2['WLAN'].str.split('b',expand = True)


# In[286]:


wifi[0].unique()


# In[287]:


wifi1 = wifi[0].str.split('hotspot',expand=True)


# In[288]:


wifi1[0].unique()


# In[289]:


wifi1[0].replace({'Yes/ ':'Wi-Fi Direct'},inplace = True)


# In[290]:


wifi1[0].unique()


# In[291]:


wifi2 = wifi1[0].str.split('802.11',expand=True)


# In[292]:


wifi2[0].unique()


# In[293]:


wifi2[0].replace({'Yes/ Wi-Fi Direct/ ':'Wi-Fi Direct','Yes/ Wi-Fi Direct':'Wi-Fi Direct','Yes/ Wi-Fi ':'Wi-Fi '},inplace=True)


# In[294]:


wifi2[0].unique()


# In[295]:


wifi2[0].isnull().sum()


# In[296]:


wifi2[0].fillna(wifi[0].mode()[0],inplace=True)


# In[297]:


wifi2[0].isna().sum()


# In[298]:


wifi2[0]= encode(wifi[0])


# In[299]:


wifi_new = pd.DataFrame(wifi2[0])


# In[300]:


df2['WLAN'] = wifi_new


# In[301]:


df2.info()


# ### approx_price_EUR 

# In[302]:


df2['approx_price_EUR'].unique()


# In[303]:


df3 = df2[df2['approx_price_EUR']!='Black']


# In[304]:


df3['approx_price_EUR'].unique()


# In[305]:


train = df3[df3['approx_price_EUR'].isnull()==False]


# In[306]:


train.isnull().sum()


# In[307]:


test = df3[df3['approx_price_EUR'].isnull()==True]


# In[308]:


test.info()


# In[309]:


train['approx_price_EUR'] = train['approx_price_EUR'].astype('int')


# In[310]:


import statsmodels.formula.api as smf
model = smf.ols(formula='approx_price_EUR~brand+network1+network2+network3+network4+network5+announced+status+weight_g+SIM+display_type+display_resolution+CPU+memory_card+internal_memory+RAM+primary_camera+secondary_camera+audio_jack+WLAN+NFC+USB+sensors',data=train).fit()
model.summary()


# In[311]:


plt.figure(figsize=(18,15))
sns.heatmap(train.corr(),annot=True, cmap = 'YlGnBu');


# In[312]:


train.columns


# In[313]:


X = train.drop(['approx_price_EUR'],axis =1)


# In[314]:


X.columns


# In[315]:


y = train[['approx_price_EUR']]


# In[316]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=777)


# In[317]:


lr.fit(X_train,y_train)


# In[318]:


ytrain_pre = pd.DataFrame(lr.predict(X_train))


# In[319]:


ytrain_pre


# In[320]:


r2_score(y_train,ytrain_pre)


# In[321]:


ytest_pre = pd.DataFrame(lr.predict(X_test))


# In[322]:


r2_score(y_test,ytest_pre)


# In[323]:


mse_test = mean_squared_error(y_test,ytest_pre)


# In[324]:


rmse = np.sqrt(mse_test)
rmse


# In[325]:


mse_train = mean_squared_error(y_train,ytrain_pre)


# In[326]:


rmse = np.sqrt(mse_train)
rmse


# In[327]:


clf = xgb.XGBClassifier()


# In[328]:


from sklearn.linear_model import ridge_regression


# In[329]:


l = Lasso()


# In[330]:


l.fit(X_train,y_train)


# In[331]:


l.predict(X_test)


# In[332]:


l.score(X_train,y_train)


# In[333]:


xg = xgb.XGBRegressor()


# In[334]:


xg.fit(X_train,y_train)


# In[335]:


xg.score(X_train,y_train)


# In[336]:


a=  xg.predict(X_test)


# In[337]:


xg.score(X_test,a)

