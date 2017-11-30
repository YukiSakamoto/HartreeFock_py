import math
import numpy

def factorial(n):
    if n <= 1:  return 1
    else:       return n * factorial(n - 1)

def factorial2(i):
    if i < 1:
        return 1
    else:
        return i*factorial2(i-2)

def binomial_prefactor(exponent, pow1, pow2, rx1, rx2):
    #   In 
    #       (x + cons1) ^ pow1 * (x + cons2) ^ pow2 ,
    #   return the coefficient of the [exponent]-degree term
    s = 0.
    for i in xrange(1 + exponent):
        j = exponent - i
        if i <= pow1 and j <= pow2:
            s += binomial(pow1, i) * math.pow(rx1, pow1-i) * binomial(pow2, j) * math.pow(rx2, pow2-j)
    return s

def norm_squared(pos1,pos2):
    return sum( (pos1-pos2)**2 )
