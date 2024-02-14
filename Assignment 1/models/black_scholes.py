"""
Black scholes model
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
        -   auto = Compute euler,exact method and values. True as default
    """

    def __init__(self, S,r,vol, T,N, K,auto = True):

        self.S = S
        self.r = r
        self.vol = vol
        self.T = T
        self.N = N
        self.dt = T/N
        self.K = K

        if auto == True:
            self.black_scholes_euler()
            self.black_scholes_exact()
            self.black_scholes_expectation(mode = "euler")
            self.black_scholes_expectation(mode = "exact")


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
            
    def black_scholes_expectation(self,mode = "exact"):
        """
        Expected value of an European price call option written on an asset in the Black-scholes model
    
        Inputs:
            - K = Strike price
            - mode = If we want to use the exact or euler method for St
        """

        if mode == "euler":
            if hasattr(self,'eu_St'):
                self.eu_Vt = self.option_prices(self.eu_St)
            else:
                self.black_scholes_euler()
                self.eu_Vt = self.option_prices(self.eu_St)

        elif mode == "exact":
            if hasattr(self, 'ex_St'):
                self.ex_Vt = self.option_prices(self.ex_St)
                
            else:
                self.black_scholes_exact()
                self.ex_Vt = self.option_prices(self.ex_St)
     

    def option_prices(self,St_val):
        """
        Computes the expected price fo european call option
        """

        Vt = np.zeros(self.N+1)

        for m,St in enumerate(St_val):
            tao = self.T - m*self.dt
            d1 = (np.log(St/self.K) +  (self.r + 0.5*(self.vol**2))*(tao))/(self.vol*np.sqrt(tao))

            d2 = d1 - self.vol*np.sqrt(tao)
            Vt[m] = St*norm.cdf(d1) - np.exp(-self.r*tao)*self.K*norm.cdf(d2)
            if m == 0:
                self.delta = norm.cdf(d1)
        
        return Vt


if __name__ == "__main__":

    vol = 0.2
    S = 100
    T = 1.
    N = 50
    r = 0.06
    K = 99
    black_scholes = Black_scholes(S,r,vol,T,N,K)
    v_black_scholes = black_scholes.ex_Vt

    s = black_scholes.ex_St
    #print(s)
    print(v_black_scholes[0])
    """
    plt.plot(vols,v_black_scholes, label = "Black scholes")
    plt.title("Call value")
    plt.legend()
    plt.show()
    """
    

