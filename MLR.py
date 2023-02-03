import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MLR():
    def __init__(self):
        self.X=None
        self.y=None
        self.W=None
        self.w=None
        self.b=0
        
    def fit(self,X,y):
        """
        Uses Linear Regression to fit X and y
        Arguments:
            X: DataFrame 
            y: Array
        """
        X=pd.DataFrame(X)
        self.X=X
        self.y=y
        self.W=np.zeros(X.shape[1])
        self.w=X.shape[1]
        X_norm = X - X.mean()
        y_norm = y - y.mean()
        xm=np.zeros((self.w,self.w))
        ym=np.zeros(self.w)
        for i in range(self.w):
            xm[i]=X_norm.multiply(X.iloc[:,i], axis="index").sum()
            ym[i]=(X.iloc[:,i]*y_norm).sum()

        self.W=np.linalg.solve(xm,ym)
        print("\nWeights:    ",self.W)
        self.b=y.mean()-(self.X.mean()*self.W).sum()
        print("Bias:       ",self.b)
        print("MSE error:  ", self.mse(self.X,self.y))
        print("RMSE error: ", self.rmse(self.X,self.y))
        print("R^2 error:  ", self.r2(self.X,self.y))
        
    def r2(self,X,y):
        y_pred=self.predict(X)
        return 1-((((y - y_pred)**2).sum())/((y-y.mean())**2).sum())
        
    def mse(self,X,y):
        y_pred=self.predict(X)
        return((y-y_pred)**2).sum()/len(y)
    
    def rmse(self,X,y):
        return np.sqrt(self.mse(X,y))
    
    def predict(self,X):
        """
        Predicts value of X
        Arguments:
            X: Numpy Array
        Returns:
            y: Array 
        """
        return ((X*self.W).sum(axis=1)+self.b)