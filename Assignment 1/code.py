import numpy as np
import matplotlib.pyplot as plt

def buildTree(S,vol,T,N):
    dt = T/N
    matrix = np.zeros((N+1,N+1))
    
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    matrix[0,0] = S
    
    # Iterate over the lower triangle
    for i in np.arange(N+1) : # i t e r a t e o ve r rows
        for j in np.arange(i+1) : # i t e r a t e o ve r columns
        # Hint : express each cell as a combination of up
        # and down moves
            matrix[i,j] = S*(u**(i-j))*(d**j)

    return matrix

# Executing buildTree function
sigma = 0.1
S = 80
T = 1.
N = 2
buildTree(S,sigma,T,N)


def valueOptionMatrix(tree , T, r, K, vol):
    dt =T/N
    u= np.exp(vol*np.sqrt(dt))
    d= np.exp(-vol*np.sqrt(dt))
    p= (np.exp(r*dt) - d)/(u-d)    
    columns = tree.shape[1]
    rows = tree.shape[0]

    #Walk backward, we start in last row of the matrix
    #Add the payoff function in the last row

    for c in np.arange(columns):
        S = tree[rows - 1, c] 
        tree[rows - 1, c] = np.maximum(0.,  S - K)


    #For all other rows, we need to combine from previous rows
    #We walk backwards, from the last row to the first row

    for i in np.arange(rows - 1)[:: -1]:
        for j in np.arange(i + 1):
            down= tree[ i + 1, j ]
            up= tree[ i + 1, j + 1]
            tree[i , j ] = np.exp(-r*dt)*(p*up + (1-p)*down)
    print(tree)
    return tree



# Executing code
sigma = 0.1
S = 80
T = 1.
N = 5
K = 85
r = 0.1
tree = buildTree(S,sigma,T, N)
valueOptionMatrix(tree,T,r,K,sigma)



"""
# Plotting
# Play around with different ranges of N and stepsizes.
N = np.arange(1,300)
# Calculatetheoptionpriceforthecorrectparameters
optionPriceAnalytical = 0 # TODO
#calculateoptionpricefor each n in N
for n in N:
    treeN = buildTree( . . . ) # TODO
    priceApproximatedly = valueOption( . . . ) # TODO
    
# usematplotlibtoplottheanalyticalvalue
# and t h e approximated v a l u e f o r each n
"""