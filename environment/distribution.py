from typing import Callable, List, Tuple
import numpy as np
from scipy.stats import norm,lognorm,chi2,beta,uniform
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

    @classmethod
    def from_sampler(self,b: int,sampler: Callable[[],float]):
        # Generating data from a distribution
        data = [ sampler() for _ in range(100000) ]
        return self.from_data(b,data)

    @classmethod
    def from_data(self,b: int,data):
        # Generating Quantiles from 
        quantiles = [ ( np.quantile(data,i/(b+1)) , i/(b+1) ) for i in range(1,b+1) ]
        return distribution(b,quantiles)

    def sampler(self):
        # # Sampling from a linear piecewise CDF.
        # rn = U()
        # idx = int(self.b*rn)
        # if idx == b-1:
        #     return self.quantiles[b-1]
        # else:
        #     return (self.quantiles[idx+1] - self.quantiles[idx])*(self.b*rn-idx) + self.quantiles[idx] 

        # Sampling from a Q - lognormal
        return q_log_normal(U(),self.a)