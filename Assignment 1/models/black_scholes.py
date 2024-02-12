"""
Still working on the code, not sure if algorithms are right, monday I will do some proof tests
Expectation function algorithm need to recheck
"""


import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

class Black_scholes():
    """
    Black_scholes model class.
    Inputs:
        -   S = stock price
        -   r = risk-free interest rate
        -   vol = volatility % in decimals
        -   T = Time period
        -   N = Number of steps/intervals
        -   K = Strike price
        -   auto = Compute euler and exact method, True as default
    """

    def __init__(self, S,r,vol, T,N, K,auto = True):

        self.S = S
        self.r = r
        self.vol = vol
        self.T = T
        self.N = N
        self.dt = T/N

        if auto == True:
            self.black_scholes_euler()
            self.black_scholes_exact()
            self.black_scholes_expectation(K,mode = "euler")
            self.black_scholes_expectation(K,mode = "exact")


    def black_scholes_exact(self):
        """
        Stocks price of each interval N in period T 
        using the exact solution of Black scholes
        """
        ex_St= np.zeros(self.N+1)
        ex_St[0] = self.S

        #### Begin Pre-computations
        pre_1 = (self.r-(1/2)*self.vol**2)*self.dt
        pre_2 = self.vol*np.sqrt(self.dt)
        ###### End Pre-computations

        for m in range(1,self.N+1):
            Z_m = np.random.normal(0,1,1)
            ex_St[m] = ex_St[m-1]*np.exp(pre_1 + pre_2*(Z_m))
            S_t = ex_St[m]
        
        self.ex_St = ex_St


    def black_scholes_euler(self):
        """
        Stocks price of each interval N in period T 
        using the euler approximation solution of Black scholes
        """

        eu_St = np.zeros(self.N+1)
        eu_St[0] = self.S

        #### Begin Pre-computations        
        pre_1 = self.r*self.dt
        pre_2 = self.vol*np.sqrt(self.dt)
        ###### End Pre-computations

        for m in range(1,self.N+1):
            Z_m = np.random.normal(0,1,1)
            eu_St[m] = eu_St[m-1] + eu_St[m-1]*pre_1+ eu_St[m-1]*pre_2*Z_m
        
        self.eu_St = eu_St
            
    def black_scholes_expectation(self,K, mode = "exact"):
        """
        Expected value of an European price call option written on an asset in the Black-scholes model
    
        Inputs:
            - K = Strike price
            - mode = If we want to use the exact or euler method for St
        """

        self.K = K

        if mode == "euler":
            if hasattr(self,'eu_St'):
                self.eu_Vt = self.expectation(self.eu_St)
            else:
                self.black_scholes_euler()
                self.eu_Vt = self.expectation(self.eu_St)

        elif mode == "exact":
            if hasattr(self, 'ex_St'):
                self.ex_Vt = self.expectation(self.ex_St)
                
            else:
                self.black_scholes_exact()
                self.ex_Vt = self.expectation(self.ex_St)
     

    def expectation(self,St_val):
        """
        Computes the expected price fo european call option
        """

        Vt = np.zeros(self.N+1)

        ####### Begin Pre-computations

        den_d1 = self.vol*np.sqrt(self.r)
        pre_num_d1 = (self.r+ (self.vol**2)/2)*self.dt
        pre_d2 = self.vol*np.sqrt(self.dt)
        pre_Vt = np.exp(-self.r*self.dt)*self.K

        ###### End Pre-computations
        
        for m,St in enumerate(St_val):

            d1 = (np.log(St/self.K) +  pre_num_d1)/den_d1
            d2 = d1 - pre_d2
            Vt[m] = St*norm.cdf(d1) - pre_Vt*norm.cdf(d2)
        
        return Vt


if __name__ == "__main__":
    a = Black_scholes(100,0.06,0.2,1,50,99)
    

