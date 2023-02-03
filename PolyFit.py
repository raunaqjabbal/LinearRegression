import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MLR import MLR


class PolyFit(MLR):
    def __init__(self, n=3):
        self.n=n
        self.X_original=[]
        MLR.__init__(self)
    
    def transform(self, X,n=None):
        """
        Takes X and transforms it to 
        Arguments:
        X: Numpy Array
        Returns:
        Transformed Array with Polynomial Features to n degree.
        """
        n=self.n if n==None else n
        poly_features=[]
        for i in range(1, n+1):
            poly_features.append(X**i)
        return np.column_stack(poly_features)
    
    def fit(self,X,y):
        self.X_original=X
        self.y=y
        self.X=pd.DataFrame(self.transform(X))
        super().fit(self.X,self.y)
        
    def plot_residuals(self):
        y_resid=self.residuals()
        plt.style.use('Solarize_Light2')
        plt.scatter(self.X_original,y_resid)
        plt.xlabel("X")
        plt.ylabel("Residual Values")
        plt.title("Residual plot")     
        plt.show()
        
    def plot(self):
        plt.style.use('Solarize_Light2')
        plt.scatter(self.X_original,self.y, c='red')
        left,right=plt.xlim()
        x_demo=np.linspace(left,right,50)
        plt.plot(x_demo, self.predict2(x_demo),c='blue')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Regression Plot")   
        plt.legend(["Actual Values","Predicted Curve"])  
        plt.show()
        
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