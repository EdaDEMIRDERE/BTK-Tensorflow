import numpy as np
import pandas as pd

## SERIES ##
data = [3, 8, 10, 52]
index = ["a", "a", "c", "d"]

series1 = pd.Series(data=data, index=index)
print("series\n", series1, "\n")
print(series1["a"], "\n")

## DATAFRAME ##
data_d = np.random.randn(4, 3)
dataFrame = pd.DataFrame(data_d)
print("dataframe\n", dataFrame, "\n")

new_data_d = pd.DataFrame(data_d, index=index, columns=["100", "101", "102"])
print("new dataframe\n", new_data_d, "\n")
print("column seçme: \n", new_data_d[["100", "101"]], "\n")
print("row seçme: \n", new_data_d.loc["a"], "\n")  # iloc index olarak çalışır.

## COLUMN EKLEME ##
new_data_d["x"] = new_data_d[["100"]] * 2
print("add dataframe\n", new_data_d, "\n")

## COLUMN SİLME ##
drop_column = new_data_d.drop("x", axis=1, inplace=True)  # axis 1 column, axis 0 row
print("drop_cloumn dataframe\n", new_data_d, "\n")  # drop değişkeni NONE dönüyor

## ROW SİLME ##
drop_row = new_data_d.drop("d", axis=0, inplace=True)  # inplace - işlemi işinde yap değişikliği kendinde yapar
print("drop_row dataframe\n", new_data_d, "\n")

## DEĞERE ULAŞMA ##
print("c - 101\n", new_data_d.loc["c", "101"], "\n")

## BOOL DEĞERLER ##
bool_frame = new_data_d < 0
print(bool_frame, "\n")

buyuk_mu = new_data_d[new_data_d["101"] > 0]  # burada true olanları gösterir
print(buyuk_mu, "\n")

## INDEX SIFIRLAMA - DEĞİŞTİRME ##
print(new_data_d.reset_index())

new_index_list = ["k", "l", "m"]
new_data_d["New"] = new_index_list
print(new_data_d.set_index("New"), "\n")  # "inplace=True" eklersek içerisindekiler değişir!!!

## MULTI INDEX ##

first_index = ["Simpson", "Simpson", "Simpson", "South Park", "South Park", "South Park"]
second_index = ["Homer", "Bart", "Marge", "Cartman", "Kenny", "Kyle"]
zip_index = list(zip(first_index, second_index))
print(zip_index, "\n")

merge_index = pd.MultiIndex.from_tuples(zip_index)
print(merge_index, "\n")

age_occ_list = [[40, "A"], [10, "B"], [30, "C"], [9, "D"], [10, "E"], [11, "F"]]
age_occ_list = np.array(age_occ_list)

cartoon_dataframe = pd.DataFrame(data=age_occ_list, index=merge_index, columns=["Age", "Occupation"])
cartoon_dataframe.index.names = ["Cartoon", "Name"]
print(cartoon_dataframe, "\n")

## MISSING VALUES ##

dic = {"Izmir": [40, 42, 39],
       "Ankara": [20, np.nan, 25],
       "Aydın": [30, 29, np.nan]}
weather = pd.DataFrame(dic)
print(weather, "\n")
print(weather.dropna(axis=1), "\n")  # 0 dersek satır yine - thresh=2 dersek 2 ve üstü nan ları atar kalanı gösterir

print(weather.fillna(20), "\n")  # nan gördüğü yere 20 yazdı

## GROUPBY ##

maas = {"Departman": ["Yazılım", "Yazılım", "Pazarlama", "Pazarlama", "Hukuk", "Hukuk"],
        "İsim": ["Ahmet", "Mehmet", "Ali", "Ayşe", "Fatma", "Zeynep"],
        "Maas": [100, 150, 200, 300, 400, 500]}
maas_dataframe = pd.DataFrame(maas)
group = maas_dataframe.groupby("Departman")
print(group.count(), "\n")
# print(group.mean(), "\n") #TODO bu olmadı


## CONCATENATION ##

dic_1 = {"Isim": ["Ahmet", "Mehmet", "Musa"],
         "Spor": ["Koşu", "Yüzme", "Koşu"],
         "Kalori": [100, 200, 300]}
dic_1_1 = pd.DataFrame(dic_1, index=[0, 1, 2])

dic_2 = {"Isim": ["Osman", "Levent", "Zeynep"],
         "Spor": ["Basketbol", "Voleybol", "Koşu"],
         "Kalori": [150, 250, 350]}
dic_2_1 = pd.DataFrame(dic_2, index=[3, 4, 5])

dic_3 = {"Isim": ["Ayşe", "Mahmut", "Duygu"],
         "Spor": ["Futbol", "Jimnastik", "Fitness"],
         "Kalori": [400, 100, 700]}
dic_3_1 = pd.DataFrame(dic_3, index=[6, 7, 8])

concat = pd.concat([dic_1_1, dic_2_1, dic_3_1])  # axis = 0
print("concat\n", concat, "\n")

## MERGE ##

m_dic_1 = {"Isim": ["Ahmet", "Mehmet", "Musa"],
           "Spor": ["Koşu", "Yüzme", "Koşu"]
           }
m_dic_2 = {"Isim": ["Ahmet", "Mehmet", "Musa"],
           "Kalori": [100, 200, 300]
           }

m_dic_11 = pd.DataFrame(m_dic_1)
m_dic_22 = pd.DataFrame(m_dic_2)

merge = pd.merge(m_dic_11, m_dic_22, on="Isim")
print("merge\n", merge, "\n")

##  ##
print(m_dic_11["Spor"].unique(), "\n")  # kaç farklı olduğu isimleri
print(m_dic_11["Spor"].nunique(), "\n")  # kaç farklı olduğu tane
print(m_dic_11["Spor"].value_counts(), "\n")  # kaç farklı olduğu neyden


def net_kalori(kalori):
    return kalori - 20


print(m_dic_22["Kalori"].apply(net_kalori), "\n")  # fonksiyonu uygular

print(m_dic_11.isnull(), "\n")  # null var mı yok mu

cd = {"Karakter Sınıfı": ["SP", "S", "SP", "S"],
      "Karakter Ismi": ["Cartman", "Kenney", "Homer", "Bart"],
      "Karakter Yas": [10, 11, 12, 13]}
cd = pd.DataFrame(cd)
cd = cd.pivot_table(values="Karakter Yas", index=["Karakter Sınıfı", "Karakter Ismi"])
print(cd, "\n")

## EXCELL ##

df = pd.read_excel(...)
df.dropna()
df.to_excel(...) #excele yazdır
