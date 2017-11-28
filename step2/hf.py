import math
import numpy as np

from math_util import *
from gto_eval import *

class PrimitiveGTO:
    def __init__(self, lmn, origin, exponent):
        self.lmn = lmn
        self.exponent = exponent
        self.origin = np.zeros(3)
        for i in range(3):
            self.origin[i] = origin[i]
        self.nc = self.calculate_normalization_factor()  # nc: Normalization Constant
    def calculate_normalization_factor(self):
        (l,m,n) = self.lmn
        alpha = self.exponent
        n = math.sqrt( 
                (math.pow(2,2*(l+m+n)+1.5)*math.pow(alpha,l+m+n+1.5)) \
                        / (factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1)*math.pow(math.pi,1.5)) )
        return n

class ContractedGTO:
    def __init__(self, lmn, origin, exponent_list, coefficient_list):
        if len(coefficient_list) != len(exponent_list): 
            raise
        self.pgto_list = []
        n_contraction = len(exponent_list)
        for i in range(n_contraction):
            self.pgto_list.append( (coefficient_list[i],  PrimitiveGTO(lmn, origin, exponent_list[i])) )
        self.nc = self.calculate_normlization_factor()  # nc: Normalization Constant
    def calculate_normlization_factor(self):
        s = 0.
        for (coeff1, pgto1) in self.pgto_list:
            for (coeff2, pgto2) in self.pgto_list:
                s += coeff1 * coeff2 * overlap_PGTO(pgto1, pgto2)
        return 1./math.sqrt(s)
    def __len__(self):
        return len(self.pgto_list)
    def __getitem__(self, i):
        return self.pgto_list[i]

class Nuclear:
    def __init__(self,pos, atomic_number):
        self.pos = np.zeros(3)
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]
        self.pos[2] = pos[2]
        self.atomic_number = atomic_number

def calc_S(basis_array):
    dim = len(basis_array)
    S = np.zeros( (dim, dim) )
    for i in range(dim):
        for j in range(dim):
            S[i,j] = overlap_CGTO(basis_array[i], basis_array[j])
    return S

def calc_T(basis_array):
    # Kinetic Energy
    dim = len(basis_array)
    T = np.zeros( (dim, dim) )
    for i in range(dim):
        for j in range(dim):
            T[i,j] = kinetic_CGTO(basis_array[i], basis_array[j])
    return T

def calc_K(basis_array, atom_list):
    dim = len(basis_array)
    V = np.zeros( (dim,dim) )
    for atom in atom_list:
        _v = np.zeros( (dim, dim) )
        for i in range(dim):
            for j in range(dim):
                _v[i,j] = nuc_attr_CGTO(basis_array[i], basis_array[j], atom)
        V += _v
    return V

def calc_G(D,basis_array):
    # Szabo., pp141(3.154)
    dim = len(basis_array)
    G = np.zeros((dim,dim))
    for u in range(dim):
        for v in range(dim):
            temp = 0.
            for p in range(dim):
                for q in range(dim):
                    doubleJ = elec_repulsion_CGTO(basis_array[u],basis_array[v],basis_array[p],basis_array[q])
                    K = 0.5 * elec_repulsion_CGTO(basis_array[u],basis_array[q],basis_array[p],basis_array[v])
                    temp += D[p,q] * (doubleJ - K)
            G[u,v] = temp
    return G

def guess_D():
    #dim = len(basis_array)
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

def rhf(nelec, basis_functions, nuclear_list):
    n_occ_orbitals = int(nelec/2)
    max_iteration = 10
    convergence_threshold = 1.e-4
    dim = len(basis_functions) # The number of Basis functions

    nuclear_repulsion = 0.
    for i in range(len(nuclear_list)):
        for j in range(i+1,len(nuclear_list)):
            norm = np.linalg.norm(nuclear_list[i].pos - nuclear_list[j].pos)
            nuclear_repulsion += nuclear_list[i].atomic_number * nuclear_list[j].atomic_number / norm

    S = calc_S(basis_functions)
    T = calc_T(basis_functions)
    K = calc_K(basis_functions, nuclear_list)
    Hcore = T+K

    #X = symmetric_orthogonalization(S)
    X = canonical_orthogonalization(S)
    X_adj = np.matrix.getH(X)  # np.matrix.getH() returns the adjoint matrix

    assert np.allclose( X_adj.dot(S.dot(X)), np.identity(dim) ), \
            "X does not satisfy (X_adjoint * S * X)"

    D = guess_D()

    # enter the SCF loop
    for i in range(max_iteration):
        print("**** iteration: {} ****".format(i+1))
        G  = calc_G(D,basis_functions)
        F = Hcore + G
        print("Fock Matrix {}".format(i+1))
        print(F)

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

        Etot = E0+nuclear_repulsion
        print("E0:  ", E0)
        print("Etot:", Etot)
        print("RMSD:", rmsd)
        if rmsd < convergence_threshold:
            break
    return Etot

def H2_1_4_eri_test():
    def calc_S_H2():
        S = np.zeros( (2,2) )
        S[0, 0] = S[1,1] = 1.
        S[0, 1] = S[1,0] = 0.6593
        return S
    def calc_Hcore_H2():
        Hcore = np.zeros( (2,2) )
        Hcore[0,0] = Hcore[1,1] = -1.1204
        Hcore[0,1] = Hcore[1,0] = -0.9584
        return Hcore

    def eri_table(u,v,p,q):
        if u < v: u,v = v,u
        if p < q: p,q = q,p
        if (u+1)*(v+1) < (p+1)*(q+1): u,v,p,q = p,q,u,v
        # Szabo. pp162(p3.235)
        if (u,v,p,q) == (0,0,0,0) or (u,v,p,q) == (1,1,1,1):    return 0.7746
        elif (u,v,p,q) == (1,1,0,0):                            return 0.5697
        elif (u,v,p,q) == (1,0,0,0) or (u,v,p,q) == (1,1,1,0):  return 0.4441
        elif (u,v,p,q) == (1,0,1,0):                            return 0.2970
        else: # Never get here.
            print(u,v,p,q)
            raise

    nuclear = [ Nuclear([0,  0, 0], 1), Nuclear([1.4,0, 0], 1)]
    bfs = []
    bfs.append( ContractedGTO( (0,0,0), [0,  0,0], [0.168856, 0.623913, 3.42525], [0.444635, 0.535328, 0.154329] ) )
    bfs.append( ContractedGTO( (0,0,0), [1.4,0,0], [0.168856, 0.623913, 3.42525], [0.444635, 0.535328, 0.154329] ) )
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    assert (abs(eri_table(i,j,k,l) - elec_repulsion_CGTO(bfs[i],bfs[j],bfs[k],bfs[l])) < 0.0001), \
                            "two electron integrals error too large"
    
    assert np.allclose(calc_S_H2(), calc_S(bfs), atol=1e-04), "Overlap Integral Incorrect"
    assert np.allclose(calc_Hcore_H2(), calc_K(bfs,nuclear)+calc_T(bfs), atol=1e-04), "Overlap Integral Incorrect"
    print("H2: Integral Check passed")


def H2(d):
    # Argument d: distance between nuclears
    nuclear = []
    nuclear.append(Nuclear([0,0,0], 1))
    nuclear.append(Nuclear([d,0,0], 1))

    basis_array_H2 = []
    basis_array_H2.append( ContractedGTO( (0,0,0), [0,0,0], [0.168856, 0.623913, 3.42525], [0.444635, 0.535328, 0.154329] ) )
    basis_array_H2.append( ContractedGTO( (0,0,0), [d,0,0], [0.168856, 0.623913, 3.42525], [0.444635, 0.535328, 0.154329] ) )
    E0 = rhf(2, basis_array_H2, nuclear)
    return E0

if __name__ == "__main__":
    H2_1_4_eri_test()
    print( H2(1.4) )
