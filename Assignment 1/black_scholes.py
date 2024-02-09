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
    
def buildTree(S,vol,T,N):
    dt = T/N
    matrix = np.zeros((N+1,N+1))
    
    u= np.exp(vol*np.sqrt(dt))
    d= np.exp(-vol*np.sqrt(dt))
    matrix[0,0] = S
    
    # Iterate over the lower triangle
    for i in np.arange (N + 1) : # i t e r a t e o ve r rows
        print(f" i is:{i}")
        for j in np.arange (i + 1) : # i t e r a t e o ve r columns
    
        # Hint : express each cell as a combination of up
        # and down moves
            print(f" j is:{j}")

    return matrix

    
if __name__ == "__main__":
    S = 100
    r = 0.06
    K = 99
    N = 3
    vol = 0.2
    T = 1
    buildTree(S,vol,T,N)