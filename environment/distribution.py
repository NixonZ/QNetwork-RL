from typing import Callable, List, Tuple
import numpy as np
from scipy.stats import lognorm
from random import random

U = lambda : random()
Exp = lambda lmbda: -1.0*np.log(U())/lmbda

def q_log_normal(y,a):
        return a[0] + a[1]*lognorm.ppf(y,1) + a[2]*y*lognorm.ppf(y,1) + a[3]*y

class distribution():

    def __init__(self,b: int,quantiles: List[Tuple]):
        # Properties of a general distribution

        self.b = b # no. of quantiles
        self.quantiles = quantiles # List of quantiles

        assert( len(quantiles) == b )

        # Converting quantiles into a QPD.
        Y = []
        X = []

        for quantile in quantiles:
            X.append(quantile[0])
            row = [1]
            row.append( lognorm.ppf(quantile[1],1) )
            row.append( quantile[1]*lognorm.ppf(quantile[1],1) )
            row.append( quantile[1] )
            Y.append(row)

        X = np.array(X)
        Y = np.array(Y)
        temp = np.dot( np.linalg.inv(np.dot(Y.T,Y)) , Y.T)
        self.a = np.dot(temp,X)

    @property
    def quantile_val(self):
        return [quantile[0] for quantile in self.quantiles]

    @classmethod
    def from_sampler(self,b: int,sampler: Callable[[],float]):
        # Generating data from a distribution
        data = [ sampler() for _ in range(10000) ]
        return self.from_data(b,data)

    @classmethod
    def from_data(self,b: int,data):
        # Generating Quantiles from 
        quantiles = [ ( np.quantile(data,i/(b+1)) , i/(b+1) ) for i in range(1,b+1) ]
        return distribution(b,quantiles)

    def sampler(self,kind = 'QPD'):
        # Sampling from a linear piecewise CDF.
        if kind == "piecewise linear":
            rn = U()
            idx = int(self.b*rn)
            if idx == self.b-1:
                return self.quantiles[self.b-1][0]
            else:
                return (self.quantiles[idx+1][0] - self.quantiles[idx][0])*(self.b*rn-idx) + self.quantiles[idx][0]

        # Sampling from a Q - lognormal
        elif kind == "QPD":
            return q_log_normal(U(),self.a)
    
    def distance(self,dist):
        assert(self.b == dist.b)
        distance = 0.0
        for i in range(self.b):
            temp = self.quantile_val[i] - dist.quantile_val[i]
            distance += np.abs(temp)
        return distance/self.b