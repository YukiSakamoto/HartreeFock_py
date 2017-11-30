import numpy as np
import math
from math_util import *

def product_GTO_center(pgto1, pgto2):
    gamma = pgto1.exponent + pgto2.exponent
    return (pgto1.exponent * pgto1.origin + pgto2.exponent * pgto2.origin) / gamma

def binomial(n, m):
    # return the nCm
    return factorial(n) / factorial(m) / factorial(n-m)

def binomial_prefactor(exponent, pow1, pow2, rx1, rx2):
    #  (x + const1) ^ pow1 * (x + const2) ^ pow2 ,
    #   return the coefficient of the [exponent]-degree term
    s = 0.
    for i in range(1 + exponent):
        j = exponent - i
        if i <= pow1 and j <= pow2:
            s += binomial(pow1, i) * math.pow(rx1, pow1-i) * binomial(pow2, j) * math.pow(rx2, pow2-j)
    return s

def boys(n, x):
    import scipy.special as sp
    # x -> 0: the denominator goes to 0, zero deviation error will occur.
    # thus, have to return 0-limit value(x->0.)
    if x < 0.: raise
    radius = 0.0001
    if x < radius:  return 1. / (2.0 * n + 1.0)
    else:
        numerator =  sp.gammainc(n+0.5, x)  * sp.gamma(n + 0.5)
        denominator = 2.0 * math.pow(x, n + 0.5)
        return numerator /denominator

def _overlap3D(lmn1,exp1,center1, lmn2,exp2,center2):
    def _overlap_1d(dim):
        s = 0.
        l1 = lmn1[dim]
        l2 = lmn2[dim]
        pa_ = pa[dim]
        pb_ = pb[dim]
        for i in range(1 + int(math.floor((l1+l2)/2))):
            s += binomial_prefactor(2*i, l1, l2, pa_, pb_) * factorial2(2*i-1) / math.pow(2*gamma, i)
        return s
    distance = np.linalg.norm(center1 - center2)
    gamma = exp1 + exp2
    Rp = (exp1*center1 + exp2*center2) / gamma
    pa = Rp-center1
    pb = Rp-center2
    prefactor = math.pow(math.pi/gamma, 1.5) * math.exp(-exp1*exp2*math.pow(distance,2)/gamma)
    sx = _overlap_1d(0)
    sy = _overlap_1d(1)
    sz = _overlap_1d(2)
    return prefactor * sx * sy * sz

def overlap_PGTO(pgto1, pgto2):
    s = _overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, pgto2.lmn, pgto2.exponent, pgto2.origin)
    return  s * pgto1.nc * pgto2.nc

def kinetic_PGTO(pgto1, pgto2):
    exp1, exp2 = pgto1.exponent, pgto2.exponent
    Ra, Rb = pgto1.origin, pgto2.origin
    (l1, m1, n1) = pgto1.lmn
    (l2, m2, n2) = pgto2.lmn
    term1 = exp2 * (2*(l2+m2+n2)+3) * _overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, pgto2.lmn, pgto2.exponent, pgto2.origin)
    term2 = -2 * math.pow(exp2,2) * (\
            _overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2+2,m2,n2), pgto2.exponent, pgto2.origin) + 
            _overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2,m2+2,n2), pgto2.exponent, pgto2.origin) + 
            _overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2,m2,n2+2), pgto2.exponent, pgto2.origin) )
    term3 = -0.5*(\
            l2*(l2-1)*_overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2-2,m2,n2), pgto2.exponent, pgto2.origin) + 
            m2*(m2-1)*_overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2,m2-2,n2), pgto2.exponent, pgto2.origin) + 
            n2*(n2-1)*_overlap3D(pgto1.lmn, pgto1.exponent, pgto1.origin, (l2,m2,n2-2), pgto2.exponent, pgto2.origin) )
    return (term1+term2+term3)*pgto1.nc*pgto2.nc

def nuc_attr_PGTO(pgto1, pgto2, nuc_pos):
    def _g_list(dim): #dim: 0,1 and 2 correspond to x,y and z, respectively
        l1 = pgto1.lmn[dim]
        l2 = pgto2.lmn[dim]
        g_list = np.zeros(l1 + l2 + 1)
        for i in range(1+l1+l2):
            for r in range(1+int(math.floor(i/2.0))):
                for u in range(1+int(math.floor((1-2*r)/2))):
                    I = i - 2*r - u
                    term1 = math.pow(-1,i) * binomial_prefactor(i,l1,l2,PA[dim],PB[dim])
                    term2_numerator = math.pow(-1,u) * factorial(i) * \
                            math.pow(CP[dim], i-2*r-2*u) * math.pow(0.25/gamma, r+u)
                    term2_denominator = factorial(r) * factorial(u) * factorial(i-2*r-2*u)
                    g_list[I] += term1 * term2_numerator / term2_denominator
        return g_list
    exp1, exp2 = pgto1.exponent, pgto2.exponent
    gamma = exp1+exp2
    Ra, Rb = pgto1.origin, pgto2.origin
    lmn1 = pgto1.lmn
    lmn2 = pgto2.lmn
    dist2 = norm_squared(pgto1.origin, pgto2.origin)     # norm * norm
    Rp = product_GTO_center(pgto1, pgto2)
    pc2 = norm_squared(nuc_pos, Rp)
    pre = 2.0 * math.pi / gamma * math.exp(-exp1 * exp2 * dist2 / gamma)

    PA = Rp - pgto1.origin
    PB = Rp - pgto2.origin
    CP = Rp - nuc_pos
    g_list_x = _g_list(0)
    g_list_y = _g_list(1)
    g_list_z = _g_list(2)
    s = 0. 
    for I in range(1+lmn1[0]+lmn2[0]):
        for J in range(1+lmn1[1]+lmn2[1]):
            for K in range(1+lmn1[2]+lmn2[2]):
                s += g_list_x[I] * g_list_y[J] * g_list_z[K] * boys(I+J+K,gamma*pc2)
    s = s * pre * pgto1.nc * pgto2.nc
    return s

def elec_repulsion_PGTO(pgto1, pgto2, pgto3, pgto4):
    def _c_list(dim):
        c_list = np.zeros(lmn1[dim] + lmn2[dim] + lmn3[dim] + lmn4[dim] + 1)
        for i1 in range(1+lmn1[dim]+lmn2[dim]):
            for i2 in range(1+lmn3[dim]+lmn4[dim]):
                for r1 in range(1+int(math.floor(i1/2))):
                    for r2 in range(1+int(math.floor(i2/2))):
                        for u in range( int(math.floor(i1+i2/2)-r1-r2+1) ):
                            f1 = binomial_prefactor(i1,lmn1[dim],lmn2[dim],PA[dim],PB[dim]) * factorial(i1) * math.pow(4*gamma1,r1) / (math.pow(4*gamma1,i1)*factorial(r1) * factorial(i1-2*r1))
                            f2 = binomial_prefactor(i2,lmn3[dim],lmn4[dim],QC[dim],QD[dim]) * factorial(i2) * math.pow(4*gamma2,r2) / (math.pow(4*gamma2,i2)*factorial(r2) * factorial(i2-2*r2))
                            f3_numerator = factorial(i1+i2-2*(r1+r2)) *math.pow(-1,u) * math.pow((q-p)[dim], i1+i2-2*(r1+r2)-2*u)
                            f3_denominator = factorial(u)*factorial(i1+i2-2*(r1+r2)-2*u) * math.pow(delta,i1+i2-2*(r1+r2)-u)
                            I = i1+i2-2*(r1+r2)-u
                            c_list[I] += f1*math.pow(-1,i2)*f2*f3_numerator/f3_denominator
        return c_list
    (lmn1,lmn2,lmn3,lmn4) = (pgto1.lmn,pgto2.lmn,pgto3.lmn,pgto4.lmn)
    (exp1,exp2,exp3,exp4) = (pgto1.exponent,pgto2.exponent,pgto3.exponent,pgto4.exponent)
    gamma1 = pgto1.exponent + pgto2.exponent
    gamma2 = pgto3.exponent + pgto4.exponent
    p = product_GTO_center(pgto1,pgto2)
    q = product_GTO_center(pgto3,pgto4)
    Rab2 = norm_squared(pgto1.origin, pgto2.origin)
    Rcd2 = norm_squared(pgto3.origin, pgto4.origin)
    Rpq2 = norm_squared(p,q)
    PA = p-pgto1.origin
    PB = p-pgto2.origin
    QC = q-pgto3.origin
    QD = q-pgto4.origin
    delta = (1./gamma1 + 1./gamma2) / 4.
    val = 0.
    c_list_x = _c_list(0)
    c_list_y = _c_list(1)
    c_list_z = _c_list(2)
    val = 0.
    for I in range(1+lmn1[0]+lmn2[0]+lmn3[0]+lmn4[0]):
        for J in range(1+lmn1[1]+lmn2[1]+lmn3[1]+lmn4[1]):
            for K in range(1+lmn1[2]+lmn2[2]+lmn3[2]+lmn4[2]):
                val += c_list_x[I] * c_list_y[J] * c_list_z[K] * boys(I+J+K, Rpq2/(4*delta))
    val *= 2*math.pow(math.pi, 2.5) / (gamma1*gamma2) / math.pow(gamma1+gamma2, 0.5) * math.exp(-(exp1*exp2*Rab2/gamma1)-(exp3*exp4*Rcd2/gamma2))
    val *= pgto1.nc*pgto2.nc*pgto3.nc*pgto4.nc
    return val

def overlap_CGTO(cgto1, cgto2):
    val = 0.
    for i in range(len(cgto1)):
        (coeff1, pgto1) = cgto1[i]
        for j in range(len(cgto2)):
            (coeff2, pgto2) = cgto2[j]
            val += coeff1 * coeff2 * overlap_PGTO(pgto1, pgto2)
    val = val * cgto1.nc * cgto2.nc
    return val

def kinetic_CGTO(cgto1, cgto2):
    val = 0.
    for i in range(len(cgto1)):
        (coeff1, pgto1) = cgto1[i]
        for j in range(len(cgto2)):
            (coeff2, pgto2) = cgto2[j]
            val += coeff1 * coeff2 * kinetic_PGTO(pgto1, pgto2)
    val = val * cgto1.nc * cgto2.nc
    return val

def nuc_attr_CGTO(cgto1, cgto2, nuclear):
    val = 0.
    for i in range(len(cgto1)):
        (coeff1, pgto1) = cgto1[i]
        for j in range(len(cgto2)):
            (coeff2, pgto2) = cgto2[j]
            val += coeff1 * coeff2 * nuc_attr_PGTO(pgto1, pgto2, nuclear.pos)
    val = -val * cgto1.nc * cgto2.nc * nuclear.atomic_number
    return val

def elec_repulsion_CGTO(cgto1, cgto2, cgto3, cgto4):
    val = 0.
    for i in range(len(cgto1)):
        (coeff1, pgto1) = cgto1[i]
        for j in range(len(cgto2)):
            (coeff2, pgto2) = cgto2[j]
            for k in range(len(cgto3)):
                (coeff3, pgto3) = cgto3[k]
                for l in range(len(cgto4)):
                    (coeff4, pgto4) = cgto4[l]
                    val += coeff1*coeff2*coeff3*coeff4*elec_repulsion_PGTO(pgto1, pgto2, pgto3, pgto4)
    val = val * cgto1.nc * cgto2.nc * cgto3.nc * cgto4.nc
    return val
