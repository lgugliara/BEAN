import numpy as np

def wedge(A, B):
    W = np.multiply.outer(A, B) - np.multiply.outer(B, A)
    triu = np.triu_indices(A.shape[-1], k=1)
    return W[..., triu[0], triu[1]]  # d(d-1)//2 features