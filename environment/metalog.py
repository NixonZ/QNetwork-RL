from typing import Callable, List, Tuple
import numpy as np
from random import random

U = lambda : random()
Exp = lambda lmbda: -1.0*np.log(U())/lmbda

class metalog():

    def __init__(self,b: int,quantiles: List[Tuple], n_terms: int = 15,bounds:Tuple[float,float] = (-np.inf,np.inf)):
        # Properties of a meta-logistic distribution

        self.b = b # no. of quantiles
        self.quantiles = quantiles # List of quantiles
        self.n_terms = n_terms
        assert( self.n_terms >= 2 )
        assert( len(quantiles) == b )

        kind = None

        if bounds[0] >= bounds[1]:
            raise Exception("Lower bound cannot be greater or equal to Upper bound!")

        if np.isneginf(bounds[0]):
            if np.isposinf(bounds[1]):
                kind = 'unbounded'
            elif np.isneginf(bounds[1]):
                raise Exception("Upper bound cannot be negative infinity!")
            else:
                kind = 'upperbounded'
        elif np.isposinf(bounds[0]):
            raise Exception("Lower bound cannot be infinity!")
        else:
            if np.isposinf(bounds[1]):
                kind = 'lowerbounded'
            elif np.isneginf(bounds[1]):
                raise Exception("Upper bound cannot be negative infinity!")
            else:
                kind = 'bounded'

        self.kind = kind
        self.bl = bounds[0]
        self.bu = bounds[1]

        # Estimating parameters using OLS.
        Y = []
        X = []

        for quantile in quantiles:
            if self.kind == 'unbounded':
                X.append(quantile[0])
            elif self.kind == 'lowerbounded':
                X.append( np.log(quantile[0]-self.bl) )
            elif self.kind == 'upperbounded':
                X.append( -1*np.log(self.bu-quantile[0]) )
            elif self.kind == 'bounded':
                X.append( np.log( (quantile[0]-self.bl)/(self.bu-quantile[0]) ) )

            y = quantile[1]
            lny = np.log(y/(1-y))
            y_ = y - 0.5

            row = [1]
            row.append( lny )
            if self.n_terms == 2:
                Y.append(row)
                continue

            row.append( y_*lny )
            if self.n_terms == 3:
                Y.append(row)
                continue

            row.append( y_ )
            if self.n_terms == 4:
                Y.append(row)
                continue

            for i in range(5,self.n_terms+1):
                if i%2:
                    row.append( np.power( y_, (i-1)//2 ) )
                else:
                    row.append( np.power( y_, i//2-1 )*lny )

            Y.append(row)

        X = np.array(X)
        Y = np.array(Y)
        temp = np.dot( np.linalg.inv(np.dot(Y.T,Y)) , Y.T)
        self.a = np.dot(temp,X)
        self.err = np.linalg.norm( X - np.dot(Y,self.a),ord=2)

    @property
    def quantile_val(self):
        return [quantile[0] for quantile in self.quantiles]

    @classmethod
    def from_sampler(self,b: int,sampler: Callable[[],float],n_terms:int = 15,bounds:Tuple[float,float] = (-np.inf,np.inf),num_data: int = 10000):
        # Generating data from a distribution
        data = [ sampler() for _ in range(num_data) ]
        return self.from_data(b,data,n_terms,bounds)

    @classmethod
    def from_data(self,b: int,data,n_terms:int = 15,bounds:Tuple[float,float] = (-np.inf,np.inf)):
        # Generating Quantiles from 
        quantiles = [ ( np.quantile(data,i/(b+1)) , i/(b+1) ) for i in range(1,b+1) ]
        return metalog(b,quantiles,n_terms,bounds)

    def sample_transform(self,sample:float):
        if self.kind == 'unbounded':
            return sample
        elif self.kind == 'lowerbounded':
            return self.bl + np.exp(sample)
        elif self.kind == 'upperbounded':
            return self.bu - np.exp(-1*sample)
        elif self.kind == 'bounded':
            return (self.bl + self.bu*np.exp(sample))/(1+np.exp(sample))
        

    def sampler(self,kind = 'metalog'):
        # Sampling from a linear piecewise CDF.
        if kind == "piecewise linear":
            rn = U()
            idx = int(self.b*rn)
            if idx == self.b-1:
                return self.quantiles[self.b-1][0]
            else:
                return (self.quantiles[idx+1][0] - self.quantiles[idx][0])*(self.b*rn-idx) + self.quantiles[idx][0]
                
        elif kind == "metalog":
            rn = U()

            if rn == 0 and (self.kind == 'lowerbounded' or self.kind == 'bounded'):
                return self.bl
            if rn == 1 and (self.kind == 'upperbounded' or self.kind == 'bounded'):
                return self.bu

            lny = np.log(rn/(1-rn))
            y_ = rn - 0.5
            sample = 0.0
            a = self.a
            
            sample += a[0] + a[1]*lny
            if self.n_terms == 2:
                return self.sample_transform(sample)

            sample += a[2]*y_*lny
            if self.n_terms == 3:
                return self.sample_transform(sample)

            sample += a[3]*y_
            if self.n_terms == 4:
                return self.sample_transform(sample)

            for i in range(5,self.n_terms+1):
                if i%2:
                    sample += a[i-1]*np.power( y_, (i-1)//2)
                else:
                    sample += a[i-1]*np.power( y_, i//2-1 )*lny

            return self.sample_transform(sample)
    
    def distance(self,dist):
        assert(self.b == dist.b)
        distance = 0.0
        for i in range(self.b):
            temp = self.quantile_val[i] - dist.quantile_val[i]
            distance += np.abs(temp)
        return distance/self.b