import numpy as np 
import matplotlib.pyplot as plt
def black_scholes_exact(S,r,vol, T,N):
    ## We are using expected values
    
    values = np.zeros(N+1)
    dt = T/N
    S_t = S
    for m in range(N):
        Z_m = np.random.normal(0,1,1)
        values[m] = S_t*np.exp((r-(1/2)*vol**2)*dt + vol*np.sqrt(dt)*(Z_m))
        S_t = values[m]
    return values


def black_scholes_euler(S,r,vol,T,N):

    values = np.zero(N+1)
    dt = T/N
    S_t = S
    for m in range(N):
        Z_m = np.random.normal(0,1,1)
        values[m] = S_t + r+S_t*dt + vol*S_t*np.sqrt(dt)*Z_m
        S_t = values[m]
    
def black_scholes_v():

    

if __name__ == "__main__":
    S = 100
    r = 0.06
    K = 99
    N = 50
    vol = 0.2
    T = 1

    values = black_scholes_exact(S,r,vol,T,N)

    plt.plot(values)
    plt.show()    