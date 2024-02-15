import numpy as np
import matplotlib.pyplot as plt

class Binary_Tree():
    """
    Binary_tree class.
    Inputs:
        -   S = stock price
        -   r = risk-free interest rate
        -   vol = volatility % in decimals
        -   T = Time period
        -   N = Number of steps/intervals
        -   auto = Compute tree automatically, True as default
    """

    def __init__(self,S,r,vol,T,N, K,auto = True):
        
        self.S = S
        self.r = r
        self.vol = vol
        self.T = T
        self.N = N
        self.dt = T/N
        
        if auto == True:
            self.build_tree()
            self.valueOptionMatrix(K)

    def build_tree(self):
    
        matrix = np.zeros((self.N+1,self.N+1))
        
        u = np.exp(self.vol*np.sqrt(self.dt))
        d = np.exp(-self.vol*np.sqrt(self.dt))
        matrix[0,0] = self.S
        
        
        for i in np.arange(self.N+1) :
            for j in np.arange(i+1) : 

                matrix[i,j] = self.S*(u**(j))*(d**(i-j))
        
        self.tree = matrix



    def valueOptionMatrix(self,K):

        self.K = K

        columns = self.tree.shape[1]
        rows = self.tree.shape[0]
        v_tree = np.copy(self.tree)
        

        u= np.exp(self.vol*np.sqrt(self.dt))

        d= np.exp(-self.vol*np.sqrt(self.dt))

        p= (np.exp(self.r*self.dt) - d)/(u-d)   

        for c in np.arange(columns):
            St = v_tree[rows - 1, c] 
            v_tree[rows - 1, c] = np.maximum(0.,  St - self.K)


        #For all other rows, we need to combine from previous rows
        #We walk backwards, from the last row to the first row

        for i in np.arange(rows - 1)[:: -1]:
            for j in np.arange(i + 1):

                down = v_tree[ i + 1, j ]
                up = v_tree[ i + 1, j + 1]

                v_tree[i , j ] = np.exp(-self.r*self.dt)*(p*up + (1-p)*down)
       
        self.v_tree = v_tree
        self.delta = (v_tree[1,1] - v_tree[1,0])/(self.S*u - self.S*d)



        


if __name__ == "__main__":

    # Executing buildTree function
    sigma = 0.1
    S = 80
    T = 1.
    N = 100
    r = 0.1
    K = 85
    binary = Binary_Tree(S,r,sigma,T,N,K)
    print(binary.tree)
    print(binary.v_tree)
