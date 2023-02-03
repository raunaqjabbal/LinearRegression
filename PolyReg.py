import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MLR import MLR 

class PolyReg(MLR):
    def __init__(self, n=3):
        self.n=n
        self.X_original=[]
        self.indices=[]
        MLR.__init__(self)
        
    def transform(self, X):
        if self.n!=1:
            t1=[np.zeros(X.shape[0])]
            # Multiply n-1 times 
            for i in range(self.n-1):
                #a=(X)*i-1
                a=X.T if i==0 else t1.T
                t1=[]
                
                #Loop to multiply a to X 
                for i in range(X.shape[1]):
                    t1.append(a*X[:,i].T)
                
                #rearranging 
                t1=np.vstack(np.array(t1)).T
                                
            if(self.indices==[]):
                t1,self.indices= np.unique(t1,axis=1,return_index=True)
            else:
                t1=t1[:,self.indices]
            return t1
        else:
            return X
        
    def fit(self,X,y):
        self.X_original=X
        X=pd.DataFrame(self.transform(X))
        super().fit(X,y)
    
    def residuals(self):
        y_pred=self.predict(self.X)
        y_resid=self.y-y_pred
        return y_resid
    
    def predict2(self, X):
        """
        Transforms X to add polynomial features and predicts the value
        Arguments:
            X: Numpy Array
        Returns:
            y: Array
        """
        return super().predict(self.transform(X))