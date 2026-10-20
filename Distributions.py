import numpy as np
from scipy import stats
import math

class ExpDistribution():
    def __init__(self, theta, sampleSize):
        self.theta = theta
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return 1/self.theta

    def theoreticalVariance(self):
        return 1/(np.power(self.theta,2))

    def generateExpDistribution(self):
        U = stats.uniform.rvs(size=self.sampleSize)
        return (-(np.log(U)) / (self.theta))


class GammaDistribution(): 
    def __init__(self, alpha, beta, sampleSize):
        self.alpha = alpha
        self.beta = beta
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return (self.alpha / self.beta)

    def theoreticalVariance(self):
        return (self.alpha / (np.power(self.beta,2)))

    def generateGammaDistribution(self):
        try:
            Gamma = np.zeros(self.sampleSize)
            for i in range(len(Gamma)):
                U = stats.uniform.rvs(size = self.alpha)
                Gamma[i] = -(np.sum(np.log(U)))
        
            return (Gamma / self.beta)
        except TypeError:
            print("Error: alpha must be an integer")


class BetaDistribution():
    def __init__(self, alpha, beta, sampleSize):
        self.alpha = alpha
        self.beta = beta
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return (self.alpha / (self.alpha + self.beta))

    def theoreticalVariance(self):
        return ((self.alpha*self.beta) / ( (np.power(self.alpha+self.beta,2))*(self.alpha+self.beta+1)))

    def generateBetaDistribution(self):
        X = GammaDistribution(self.alpha, 1, self.sampleSize).generateGammaDistribution()
        Y = GammaDistribution(self.beta, 1, self.sampleSize).generateGammaDistribution()
        return (X/(X+Y))

class ChiSquareDistribution():
    def __init__(self, p, sampleSize):
        self.p = p
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return self.p

    def theoreticalVariance(self):
        return 2*self.p

    def generateChiSquareDistribution(self):
        return GammaDistribution(int(self.p/2), 1/2, self.sampleSize).generateGammaDistribution()


class NegBinDistribution():
    def __init__(self,r,p, sampleSize):
        self.r = r
        self.p = p
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return (self.r*(1-self.p))/(self.p)

    def theoreticalVariance(self):
        return (self.r*(1-self.p))/(np.power(self.p,2))

    def generateNegBinDistribution(self):
        NegBin = np.zeros(self.sampleSize)
        U = stats.uniform.rvs(size = self.sampleSize)


        negBinSum = self.NegBinSequence()

        for i in range(self.sampleSize):
            j = 0

            while U[i] > sum(negBinSum[0:j]):
                j+=1
            NegBin[i] = j-2

        return NegBin

    def NegBinSequence(self):
        tolerance = 1e-8
        j = 0
        negBinSum = [0]
        while sum(negBinSum) <= 1-tolerance: ## Tolerance required because sum never reaches 1, but sequence is infinite
            negBinSum.append((math.factorial(self.r+j-1)) / (math.factorial(j)*math.factorial(self.r-1))*np.power(self.p,self.r)*np.power(1-self.p,j))
            # print(NegBinSum[j])
            j+=1
        return negBinSum 

class BinDistribution():
    def __init__(self,n,p,sampleSize):
        self.n = n
        self.p = p
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return (self.n*self.p)

    def theoreticalVariance(self):
        return (self.n*self.p*(1-self.p))

    def generateBinDistribution(self):
    
        Bin = np.zeros(self.sampleSize)
        U = stats.uniform.rvs(size = self.sampleSize)
        binSum = self.BinSequence()
   

        for i in range(self.sampleSize):
            j = 0

            while U[i] > sum(binSum[:j]):
                j+=1
            Bin[i] = j-2
        return Bin

    def BinSequence(self):
        binSum = [0]
        for j in range(self.n+1):
            binSum.append( (math.factorial(self.n)/(math.factorial(j)*math.factorial(self.n-j))) * np.power(self.p,j) *np.power(1-self.p,self.n-j))
        return binSum


class NormalDistribution():
    def __init__(self,theta,sigmaSquare, sampleSize):
        self.theta = theta
        self.sigmaSquare = sigmaSquare
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return self.theta

    def theoreticalVariance(self):
        return self.sigmaSquare
    
    def generateNormalDistribution(self):
        Normal = np.zeros(self.sampleSize)

        if self.sampleSize % 2 == 1:
            raise ValueError("Please use an even Sample Size")

        U1 = stats.uniform.rvs(size = int(self.sampleSize/2))
        U2 = stats.uniform.rvs(size = int(self.sampleSize/2))
   

        for i in range(int(self.sampleSize/2)):
            Normal[2*i] = np.sqrt(-2*np.log(U1[i]))*np.cos(2*np.pi*U2[i])
            Normal[2*i+1] = np.sqrt(-2*np.log(U1[i]))*np.sin(2*np.pi*U2[i])

        return Normal*np.sqrt(self.sigmaSquare)+self.theta

class BerDistribution():
    def __init__(self, p, sampleSize):
        self.p = p
        self.sampleSize = sampleSize

    def theoreticalMean(self):
        return self.p
    def theoreticalVariance(self):
        return self.p(1-self.p)

    def generateBerDistribution(self):
        U = stats.uniform.rvs(size = self.sampleSize)
        Ber = np.zeros(self.sampleSize)

        for i in range(self.sampleSize):
            if U[i] <= self.p:
                Ber[i] = 1
            else:
                Ber[i] = 0
        return Ber

class PeakDistribution():
    def __init__(self, mu, sigma, sampleSize):
        self.p = mu
        self.d = sigma
        self.sampleSize = sampleSize

    def generatePeakDistribution(self):
        U = stats.uniform.rvs(size = self.sampleSize)
        Peak = np.zeros(self.sampleSize)
        for i in range(self.sampleSize):
            x = U[i]
            if x<=0.5:
                Peak[i] = (-1+np.sqrt((2*(x-self.mu))/self.sigma))/self.sigma
            else: 
                Peak[i] = (1-np.sqrt((2*(1-((x-self.mu)/self.sigma)))))/self.sigma
        return Peak