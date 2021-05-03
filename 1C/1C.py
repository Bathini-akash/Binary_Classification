import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fields=['X','Y','Z','A','Status']
f=open ("dataset_LP_1.txt","r")
list1=[]
for x in f:	
	y=x.rsplit(",")
	list2=[]
	list2.append(y[0])
	list2.append(y[1])
	list2.append(y[2])
	list2.append(y[3])	
	list2.append(y[4])
	list1.append(list2)
with open ("final.csv","w") as f2:
	writer=csv.writer(f2)
	writer.writerow(fields)	
	writer.writerows(list1)	
f3=pd.read_csv("final.csv")
print(f3)	
train_data=f3.sample(frac=0.5,random_state=1).reset_index(drop=True)
test_data =f3.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

rate=0.01
no_of_iterations=100
X=train_data.iloc[0:,[0,1,2,3]].values
Y=train_data.iloc[0:,4].values
w=np.zeros(train_data.shape[1])
for i in range(no_of_iterations):
	temp=0
	for x,status in zip(X,Y):
		y=np.dot(x,w[1:])+w[0]
		predict=np.where(y>= 0.0, 1, 0)
		error=rate*(status-predict)
		w[1:]+=error*x
		w[0]+=error
		if error!=0:
			temp=1
			break
	if temp==0:		
		break		
rows_in_test=test_data.shape[0] 
count=0   
for index,row in test_data.iterrows():
    x=[1,row[0],row[1],row[2],row[3]]
    y=np.dot(x,w)
    pos_neg=0# 
    if y>=0:
    	pos_neg=1
    if pos_neg==row[4]:	
    	count+=1
accuracy=(count*100)/rows_in_test
print(accuracy)    