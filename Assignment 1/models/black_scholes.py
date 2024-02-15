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
        -   S: stock price
        -   r: risk-free interest rate
        -   vol: volatility fo the stock % in decimals
        -   T: Time period
        -   N: Number of steps/intervals
        -   K: Strike price
        -   auto: Compute euler,exact method and values. True as default
    """

    def __init__(self, S,r,vol, T,N, K,auto = True):

        self.S = S
        self.r = r
        self.vol = vol
        self.T = T
        self.N = N
        self.dt = T/N
        self.K = K
        self.taos = self.T - np.arange(0,self.N+1)*self.dt

        self.delta = None
        self.deltas = None
        self.cash = None

        if auto == True:
            self.option_values(mode = "euler")
            self.option_values(mode = "exact")
            self.euler_hedging()


    def exact_method(self):
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


    def euler_method(self):
        """
        Stocks price of each interval N in period T 
        using the euler approximation solution of Black scholes
        """

        eu_St = np.zeros(self.N+1)
        eu_St[0] = self.S

        #### Begin Pre-computations        
        pre_1 = self.r*self.dt
        pre_2 = self.vol*np.sqrt(self.dt)
        #### End Pre-computations

        for m in range(1,self.N+1):
            Z_m = np.random.normal(0,1,1)
            eu_St[m] = eu_St[m-1] + eu_St[m-1]*pre_1+ eu_St[m-1]*pre_2*Z_m
        self.eu_St = eu_St
    
    def euler_hedging(self, vol_hedge = None, do_cash = False):
        """
        Hedging simulation of each interval N in period T 
        using the euler approximation solution of Black scholes
        Inputs:
            - vol_hedge: Volatility for hedge parameter computations, set as = stock volatility as default
        """

        if not hasattr(self,'eu_St'):
            self.euler_method()
        if vol_hedge == None:
            vol_hedge = self.vol
        
        ## Delta parameters
        d1s = (np.log(self.eu_St/self.K) + (self.r + 0.5*(vol_hedge**2)*self.taos))/(self.vol*np.sqrt(self.taos))
        self.deltas = norm.cdf(d1s)

        ## Cash and initial call price
        if do_cash == True:
            self.cash_hedging(vol_hedge)



    def cash_hedging(self,vol_hedge):
        """
        Simulates a how does tha account balance of an investor changes.
        Need to change to vectorized way if is necessary
        """
        f = self.eu_St[0]*self.deltas[0] - np.exp(-self.r*self.T)*self.K*norm.cdf(d1s[0] -vol_hedge*np.sqrt(self.T))
        self.cash = np.zeros(self.N + 1)
        self.cash[0] = f - self.deltas[0]*self.eu_St[0]

        ### Hedging adjustment, need to be vectorized
        for m in range(1,self.eu_St.shape[0]-1):
            cash_t = self.cash*np.exp(self.r*self.dt) - (self.deltas[m]-self.deltas[m-1])*self.eu_St[m]
            self.cash[m] = self.cash[m-1]*np.exp(self.r*self.dt) - (self.deltas[m]-self.deltas[m-1])*self.eu_St[m]
        
        #### Selling the stock at strike price at maturity
        self.cash[-1] = self.cash[-2]*np.exp(self.r*self.dt) + int(self.eu_St[-1] - self.K >0)*(self.K - (1-self.deltas[-2])*self.eu_St[-1])

    def option_values(self,mode = "exact"):
        """
        Expected value of an European price call option written on an asset in the Black-scholes model
        And hedge parameter
    
        Inputs:
            - mode = If use the exact or euler method of stock price
        """
        
        if mode == "euler":
            if not hasattr(self,'eu_St'):
                self.euler_method()
            self.eu_Vt = self.option_price(mode)

        elif mode == "exact":
            if not hasattr(self, mode):
                self.exact_method()

            self.ex_Vt = self.option_price(mode)

            ## Hedge parameter for part II question 4
            self.delta = norm.cdf((np.log(self.ex_St[0]/self.K) +  (self.r + 0.5*(self.vol**2))*(self.T))/(self.vol*np.sqrt(self.T)))

    
    def option_price(self,mode, vol_hedge = None):
        """
        Computes the expected price at time t of an european call option
        Inputs:
            - m: Position in time
            - St: Stock value
            . vol_hedge: Volatility of hedge, set as stock volatility as default
        """
        if mode == "exact":
            St = self.ex_St
        if mode == "euler":
            St = self.eu_St
        if vol_hedge == None:
            vol_hedge = self.vol

        d1s = (np.log(St/self.K) + (self.r + 0.5*(vol_hedge**2)*self.taos))/(self.vol*np.sqrt(self.taos))
        d2s = d1s - vol_hedge*np.sqrt(self.taos)
        Vt = St*norm.cdf(d1s) - np.exp(-self.r*self.taos)*self.K*norm.cdf(d2s)

        return Vt
     


if __name__ == "__main__":
    vol = 0.2
    S = 100
    T = 1.
    N = 365
    r = 0.06
    K = 99
    black_scholes = Black_scholes(S,r,vol,T,N,K, auto = True)



