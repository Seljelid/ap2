import numpy as np
import csv
import pandas
from datetime import datetime as dt

with open('../Data/ai_dataset2.csv') as file:
    df = pandas.read_csv(file,delimiter=',',low_memory='false')

#Correlation matrix
#corr = df.corr()
#print corr

#Number of NaN
print df.isnull().sum()

#Averages
#print 'Avg F1:', np.mean(df['F1'])

#Convert to numpy array
data = df.as_matrix()

#Convert date
for row in data:
    row[1] = dt.strptime(row[1],'%d%b%Y')

print data[1:10,:]
