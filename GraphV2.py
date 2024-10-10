import numpy as np
from scipy import sparse
import random

def regular_graph(n,k):
    if (n*k)%2 != 0 or n<=k:
        print('no such graph')
        return 'end'
    marbles = k*list(range(n))
    edges = random.sample(marbles,k*n)
    row = np.array([edges[i] for i in np.arange(0,len(edges),2)]+[edges[i] for i in np.arange(1,len(edges),2)])
    col = np.array([edges[i] for i in np.arange(1,len(edges),2)]+[edges[i] for i in np.arange(0,len(edges),2)])
    data = k*n*[1]
    B = sparse.coo_array((data, (row, col)), shape=(n, n)).tocsr()
    return B

def bipartite_biregular_graph(n,m,k,l):
    if (n*k) != m*l or n<k or m<l:
        print('no such graph')
        return 'end'
    marbles = l*list(range(m))
    edges = random.sample(marbles,l*m)
    row = np.array(k*list(range(n)))
    col = np.array(edges)
    data = k*n*[1]
    B = sparse.coo_array((data, (row, col)), shape=(n, m)).tocsr()
    return B

def s_regular_graph(S,N):
    k = len(N)
    blocks = []
    for i in range(k):
        block = []
        for j in range(i):
            block.append((blocks[j][i]).transpose())
        for j in range(i,k):
            if i == j:
                block.append(regular_graph(N[i],S[i,i]))
                if type(block[-1]) == str:
                    print(i,j)
                    return
            else:
                block.append(bipartite_biregular_graph(N[i],N[j],S[i,j],S[j,i]))
                if type(block[-1]) == str:
                    print(i,j)
                    return
        blocks.append(block)
    A = sparse.bmat(blocks)
    D = sparse.diags_array(1/np.sqrt(A.sum(axis=0)),A.shape)
    L = D@A@D
    return A,L

def SBM(P,N):
    k = len(N)
    blocks = []
    for i in range(k):
        block = []
        for j in range(i):
            block.append((blocks[j][i]).transpose())
        for j in range(i,k):
            N1 = N[i]
            N2 = N[j]
            rows = []
            columns = []
            temp = np.arange(1,N2+1)
            if i == j:
                for s in range(N1):
                    col = temp[(s+1):] * np.random.choice(2,N2-s-1,True,[1-P[i,j],P[i,j]])
                    col = [t-1 for t in col if t!=0]
                    columns += col
                    rows += len(col)*[s]
                data = len(columns)*[1]
                D = sparse.coo_array((data, (rows, columns)), shape=(N1, N2)).tocsr()
                block.append(D + D.transpose())
            else:
                for s in range(N1):
                    col = temp * np.random.choice(2,N2,True,[1-P[i,j],P[i,j]])
                    col = [t-1 for t in col if t!=0]
                    columns += col
                    rows += len(col)*[s]
                data = len(columns)*[1]
                block.append(sparse.coo_array((data, (rows, columns)), shape=(N1, N2)).tocsr())
        blocks.append(block)
    A = sparse.bmat(blocks)
    D = sparse.diags_array(1/np.sqrt(A.sum(axis=0)),shape=A.shape)
    L = D@A@D
    return A,L
    


