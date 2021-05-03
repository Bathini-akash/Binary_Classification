import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
f3=pd.read_csv("dataset_LP_2.csv")
print(f3)	
train_data=f3.sample(frac=0.7,random_state=1).reset_index(drop=True)
test_data =f3.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
rate=0.01
no_of_iterations=1000000
X=train_data.iloc[0:,[0,1,2]].values
Y=train_data.iloc[0:,3].values
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
	if temp==0:		
		break	

rows_in_test=test_data.shape[0] 
count=0   
for index,row in test_data.iterrows():
    x=[1,row[0],row[1],row[2]]
    y=np.dot(x,w)
    predict=0
    if y>=0:
    	predict=1
    if predict==row[3]:	
    	count+=1    	
accuracy=(count*100)/rows_in_test
print(accuracy)


#visualisation
x=np.linspace(-7.5,10,100)
y=np.linspace(-8,2,100)	
x,y=np.meshgrid(x,y)	
z=(-w[0]-w[1]*x-w[2]*y)/w[3]
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, color='yellow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
points_pos = np.where(test_data.iloc[:, -1].values == 1)
points_neg = np.where(test_data.iloc[:, -1].values == 0) 
X2 = test_data.iloc[points_pos].values[:, 0]
Y2 = test_data.iloc[points_pos].values[:, 1]
Z2 = test_data.iloc[points_pos].values[:, 2]
ax.scatter(X2, Y2, Z2, color='red')
X1 = test_data.iloc[points_neg].values[:, 0]
Y1 = test_data.iloc[points_neg].values[:, 1]
Z1 = test_data.iloc[points_neg].values[:, 2]
ax.scatter(X1, Y1, Z1, color='blue')
ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.tight_layout()
plt.show() 	