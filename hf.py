import math
import numpy as np

def calc_S_H2():
    dim = 2
    S = np.zeros( (dim, dim) )
    S[0, 0] = S[1,1] = 1.
    S[0, 1] = S[1,0] = 0.6593
    return S

def calc_Hcore_H2():
    dim = 2
    Hcore = np.zeros( (dim,dim) )
    Hcore[0,0] = Hcore[1,1] = -1.1204
    Hcore[0,1] = Hcore[1,0] = -0.9584
    return Hcore

def eri_table(u,v,p,q):
    if u < v:
        u,v = v,u
    if p < q:
        p,q = q,p
    if (u+1)*(v+1) < (p+1)*(q+1):
        u,v,p,q = p,q,u,v
    
    # Szabo. pp162(p3.235)
    if (u,v,p,q) == (0,0,0,0) or (u,v,p,q) == (1,1,1,1):
        return 0.7746
    elif (u,v,p,q) == (1,1,0,0):
        return 0.5697
    elif (u,v,p,q) == (1,0,0,0) or (u,v,p,q) == (1,1,1,0):
        return 0.4441
    elif (u,v,p,q) == (1,0,1,0):
        return 0.2970
    else:
        # Never get here.
        print(u,v,p,q)
        raise

def calc_G_H2(D):
    # Szabo., pp141(3.154)
    G = np.zeros(D.shape)
    dim = D.shape[0]
    for u in range(dim):
        for v in range(dim):
            temp = 0.
            for p in range(dim):
                for q in range(dim):
                    doubleJ = eri_table(u,v,p,q)
                    K = 0.5 * eri_table(u,q,p,v)
                    temp += D[p,q] * (doubleJ - K)
            G[u,v] = temp
    return G

def guess_D():
    dim = 2
    D = np.zeros( (dim,dim) )
    return D

def calc_D_rhf(C, n_occ_orbitals):
    #XXX for RHF only. NOT use for UHF calculation.
    #   C: new coefficient matrix
    #   n_occ_orbitals: n_electron / 2
    # Szabo. pp139 (3.145)
    row, col = C.shape
    D = np.zeros((row,col))
    for u in range(row):
        for v in range(col):
            for a in range(n_occ_orbitals):
                D[u,v] += 2. * C[u,a] * C[v,a]
    return D


def calc_E0_rhf(D,Hcore,F):
    # Szabo. pp150(3.184)
    H_F = Hcore + F
    row, col = H_F.shape
    E0 = 0.
    for u in range(row):
        for v in range(col):
            E0 += D[v,u] * H_F[u,v]
    E0 *= 0.5;
    return E0

def symmetric_orthogonalization(S):
    # Szabo. pp.143 (3.167)
    l, U = np.linalg.eigh(S)
    l_rt = np.zeros( S.shape )
    for i in range(len(l)):
        l_rt[i,i] = 1./(math.sqrt(l[i]))
    X =  U.dot(l_rt.dot(np.conjugate(U))) 
    return X

def canonical_orthogonalization(S):
    # Szabo. pp.144 (3.169 - 3.172)
    l, U = np.linalg.eigh(S)
    row,col = S.shape
    X = np.zeros(S.shape)
    for i in range(row):
        for j in range(col):
            X[i,j] = U[i,j] / (math.sqrt(l[j]))
    return X

def calc_rmsd(m1,m2):
    d = m1 - m2
    ret =  np.sum(d**2)/d.size
    return ret

def rhf(nelec):
    n_occ_orbitals = 1 # int(nelec/2)
    max_iteration = 10
    convergence_threshold = 1.e-4
    dim = 2 # The number of Basis functions

    S = calc_S_H2()
    #X = symmetric_orthogonalization(S)
    X = canonical_orthogonalization(S)
    X_adj = np.matrix.getH(X)  # np.matrix.getH() returns the adjoint matrix

    assert np.allclose( X_adj.dot(S.dot(X)), np.identity(dim) ), \
            "X does not satisfy (X_adjoint * S * X)"

    Hcore = calc_Hcore_H2()
    D = guess_D()

    # enter the SCF loop
    for i in range(max_iteration):
        print("**** iteration: {} ****".format(i+1))
        G = calc_G_H2(D)
        F = Hcore + G

        # obtain F' by rotating F matrix
        # Szabo. pp.145 (3.177)
        F_prime = X_adj.dot(F.dot(X))
        
        # Solve Eigenvalue problem
        # Szabo. pp.145 (3.178)
        e, C_new_prime = np.linalg.eigh(F_prime)

        # obtain C_new by rotating C_new_prime
        C_new = X.dot(C_new_prime)
        
        # Calculate Density Matrix
        D_new = calc_D_rhf(C_new, n_occ_orbitals)
        
        E0 = calc_E0_rhf(D, Hcore, F)
        rmsd = calc_rmsd(D_new, D)
        D = D_new
        print("E0:  ", E0)
        print("RMSD:", rmsd)
        if rmsd < convergence_threshold:
            break
    return E0

E0 = rhf(2)

assert 0.0001 < abs(E0 - 1.831), "Error is too large" 
