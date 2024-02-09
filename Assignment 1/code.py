import numpy as np

def buildTree(S,vol,T,N):
    dt = T/N
    matrix = np.zeros((N+1,N+1))
    
    u = 0 # TODO
    d = 0 # TODO
    
    # Iterate over the lower triangle
    for i in np.arange (N + 1) : # i t e r a t e o ve r rows
        for j in np.arange (i + 1) : # i t e r a t e o ve r columns
        # Hint : express each cell as a combination of up
        # and down moves
        matrix [ i , j ] = 0 # TODO
        return matrix

# Executing buildTree function
sigma = 0.1
S = 80
T = 1.
N = 2
buildTree(S,sigma,T,N)


def valueOptionMatrix(tree,T,r,K,vol) :
  dt = T/N
  u = 0 # TODO
  d = 0 # TODO
  p = 0 # TODO
  columns = tree.shape[1]
  rows = tree.shape[0]
  # Walk backward , we start in last row of the matrix
  # Add the payoff function in the last row
  for c in np.arange(columns):
    S = tree[rows − 1,c] #value in the matrix
    tree[rows − 1, c] = 0 # TODO
  # For all other rows, we need to combine from previous rows
  # We walk backwards, from the last row to the first row
  for i in np.arange(rows−1)[::−1]:
    for j in np.arange(i+1):
      down = tree[i+1,j]
      up = tree[i+1,j+1]
      tree[i,j] = 0 # TODO
  return tree

# Executing code
sigma = 0.1
S = 80
T = 1.
N = 2
K = 85
r = 0.1
tree = buildTree(S,sigma,T, N)
valueOptionMatrix(tree,T,r,K,sigma)
