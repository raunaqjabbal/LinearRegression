import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PolyFit import PolyFit
from MLR import MLR
from PolyReg import PolyReg

#####################           MLR         #########################################

# # dataset = pd.read_csv("RegressionData.csv")
# dataset=np.array([[1,0,2,21],[2,8,0,0],[0,6,1,4],[1,11,1,2]]) 
# # y=3a-b+8c+2
# dataset=pd.DataFrame(dataset)
# X = dataset.iloc[:, 0:-1].values
# y = dataset.iloc[:, -1].values
# model=MLR()
# model.fit(X,y)

#####################           PolyFit         #########################################
# dataset=np.array([[1,1],[2,8],[3,27],[4,64],[5,125]])  
# # y= a^3
# dataset=pd.DataFrame(dataset)
# X = dataset.iloc[:, 0:-1].values
# y = dataset.iloc[:, -1].values
X=np.linspace(0,2*np.pi,50)
y=np.sin(X)+0.1*np.random.randn(50)
plt.scatter(X,y,c="blue")
plt.plot(X,np.sin(X),c="red")
plt.legend(["Dataset Values","sin(x) curve"])  
plt.show()

model=PolyFit(n=1)
model.fit(X,y)
print("y_pred: ",model.predict2(X))
model.plot_residuals()
model.plot()

#####################           PolyReg         #########################################
# dataset=np.array([[1,3,10],[3,4,25],[-1,8,65],[7,7,98],[1,1,2]])  
# # y= a^2+b^2
# dataset=pd.DataFrame(dataset)
# X = dataset.iloc[:, 0:-1].values
# y = dataset.iloc[:, -1].values
# model=PolyReg(n=1)
# model.fit(X,y)
# print("y_pred: ",model.predict2(X))
# print(model.transform(X))