import numpy as np




print(np.zeros(10))

print(np.zeros((3, 4)))

print(np.ones(5))

print(np.ones((3, 5)))



print(np.eye(3))

     
print(np.array(['1.5', '0.62', '2', '3.14', '3.141592']))




str_a1 = np.array(['1.567', '0.123', '5.123', '9', '8'])
num_a1 = str_a1.astype(float)
# print(num_a1)




print(str_a1.dtype)

print(num_a1.dtype)



str_a2 = np.array(['1', '3', '5', '7', '9'])
num_a2 = str_a2.astype(int)
print(num_a2)


print(str_a2.dtype)
print(num_a2.dtype)





num_f1 = np.array([10, 21, 0.549, 4.75, 5.98])
num_i1 = num_f1.astype(int)
# print(num_i1)


print(num_f1.dtype)
print(num_i1.dtype)





print(np.random.rand(2, 3))



print(np.random.rand())



print(np.random.rand(2, 3, 4))



print(np.random.randint(10, size=(3, 4)))



print(np.random.randint(1, 30))








arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([1, 2, 3, 4])

print(arr1 + arr2)
print(arr1 - arr2)
print(arr2*2)
print(arr2**2)
print(arr1*arr2)
print(arr1/arr2)
print(arr1 / (arr2**2))
print(arr1>20)





arr3 = np.arange(5)
print(arr3)


print(arr3.sum(), arr3.mean())

print(arr3.std(), arr3.var())

print(arr3.min(), arr3.max())


arr4 = np.arange(1, 5)
print(arr4)


print(arr4.cumsum())
print(arr4.cumprod())





A = np.array([0, 1, 2, 3]).reshape(2, 2)
print(A)

B = np.array([3, 2, 0, 1]).reshape(2, 2)
print(B)


print(A.dot(B))
print(np.dot(A, B))


print(np.transpose(A))
print(A.transpose())


print(np.linalg.inv(A))

print(np.linalg.det(A))










a1 = np.array([0, 10, 20, 30, 40, 50])
print(a1)


print(a1[0])


print(a1[-2])


a1[5] = 70
print(a1)


print(a1[[1, 3, 4]])




a2 = np.arange(10, 100, 10).reshape(3, 3)
print(a2)
print(a2[0, 2])

a2[2, 2] = 95
print(a2)


print(a2[1])


a2[1] = np.array([45, 55, 65])
print(a2)



a2[1] = [47, 57, 67]
print(a2)



print(a2[[0, 2], [0, 1]]) 




'''----------------------------------------------'''





import pandas as pd



s1 = pd.Series([10, 20, 30, 40, 50])
print(s1)


print(s1.index)

print(s1.values)


s2 = pd.Series(['a', 'b', 'c', 1, 2, 3])
print(s2)


s3 = pd.Series([np.nan, 10, 30])
print(s3)


index_date = ['2025-08-07', '2025-10-08', '2025-10-09', '2025-10-10']
s4 = pd.Series([200, 195, np.nan, 205], index = index_date)
print(s4)



s5 = pd.Series({'국어': 100, '영어': 95, '수학': 90})
print(s5)





print(pd.date_range(start='2025-01-01', end='2025-01-07'))

print(pd.date_range(start='2025/01/01', end='2025.01.07'))


print(pd.date_range(start='2025-01-01', periods=7))


print(pd.date_range(start='2025-01-01', periods=4, freq='2DCX'))





index_date = pd.date_range(start= '2025-03-01', periods= 5, freq='D')
pd.Series([51, 62, 55, 49, 58], index = index_date)

print(pd.Series([51, 62, 55, 49, 58], index = index_date))




print(pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))



data_list = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(pd.DataFrame(data_list))



data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
index_date = pd.date_range('2019-09-01', periods=4)
columns_list = ['A', 'B', 'C']
print(pd.DataFrame(data, index=index_date, columns=columns_list))





table_data = {'연도': [2015, 2016, 2016, 2017, 2017],
              '지사': ['한국', '한국', '미국', '한국', '미국'],
              '고객 수': [200, 250, 450, 300, 500]}

# print(table_data)
# print(pd.DataFrame(table_data))

df = pd.DataFrame(table_data, columns=['연도', '지사', '고객 수'])
# print(df)


print(df.index)
print(df.columns)
print(df.values)








s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50])
print(s1 + s2)
print(s2 - s1)
print(s1 * s2)
print(s2 / s1)



s3 = pd.Series([1, 2, 3, 4])
s4 = pd.Series([10, 20, 30, 40, 50])

print(s3 + s4)
print(s4 - s3)
print(s3 * s4)







table_data3 = {'봄':  [256.5, 264.3, 215.9, 223.2, 312.8],
              '여름': [770.6, 567.5, 599.8, 387.1, 446.2],
              '가을': [363.5, 231.2, 293.1, 247.7, 381.6],
              '겨울': [139.3, 59.9, 76.9, 109.1, 108.1]}
columns_list = ['봄', '여름', '가을', '겨울']
index_list = ['2012', '2013', '2014', '2015', '2016']

df3 = pd.DataFrame(table_data3, columns=columns_list, index=index_list)
print(df3)


# print(df3.mean())
# print(df3.std())


print(df3.mean(axis=1))
print(df3.std(axis=1))
print(df3.describe())







KTX_data = {'경부선 KTX': [39060, 39896, 42005, 43621, 41702, 41266, 32427],
            '호남선 KTX': [7313, 6967, 6873, 6626, 8675, 10622, 9228],
            '경전선 KTX': [3627, 4168, 4088, 4424, 4606, 4984, 5570],
            '전라선 KTX': [309, 1771, 1954, 2244, 3146, 3945, 5766],
            '동해선 KTX': [np.nan, np.nan, np.nan, np.nan, 2395, 3786, 6667]}
col_list = ['경부선 KTX','호남선 KTX','경전선 KTX','전라선 KTX','동해선 KTX']
index_list = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']


df_KTX = pd.DataFrame(KTX_data, columns=col_list, index=index_list)
print(df_KTX)


print(df_KTX.index)
print(df_KTX.columns)
print(df_KTX.values)



print(df_KTX.head(1))
print(df_KTX.tail(3))


print(df_KTX[1:2])
print(df_KTX[2:5])


print(df_KTX.loc['2011'])
print(df_KTX.loc['2013':'2016'])



print(df_KTX['경부선 KTX'])


print(df_KTX['경부선 KTX']['2012':'2014'])

print(df_KTX['경부선 KTX'][2:5])


print(df_KTX.T)
print(df_KTX[['동해선 KTX','전라선 KTX','경전선 KTX','호남선 KTX','경부선 KTX']])





print(pd.read_csv('sea_rain1.csv'))
print(pd.read_csv('sea_rain1.csv', index_col="연도"))





df_WH = pd.DataFrame({'Weight': [62, 67, 55, 74],
                      'Height':[165, 177, 160, 180]},
                     index=['ID_1', 'ID_2', 'ID_3', 'ID_4'])

df_WH.index.name = 'User'
print(df_WH)

bmi = df_WH['Weight']/(df_WH['Height']/100)**2
print(bmi)

df_WH['BMI'] = bmi      # BMI 컬럼에 bmi 값을 넣는다.
print(df_WH)


df_WH.to_csv('save_DataFrame.csv')





df_pr = pd.DataFrame({'판매가격':[2000, 3000, 5000, 10000],
                      '판매량':[32, 53, 40, 25]},
                     index=['P1001', 'P1002', 'P1003', 'P1004'])
df_pr.index.name = '제품번호'
print(df_pr)




file_name = 'save_DataFrame_cp949.txt'
df_pr.to_csv(file_name, sep=" ")


tmp1 = {"a": 1, "b": 2, "c": 3}

tmp2 = {"c": 3, "d": 4, "e": 5}