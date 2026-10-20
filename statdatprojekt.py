import pandas as pd
from matplotlib import pyplot as plt
from Distributions import BetaDistribution, NormalDistribution, BinDistribution
import numpy as np
import scipy as sp

def main():
    N = 1566 ## replies
    Threshold = 6

    df = pd.read_csv("ESS11e04_2-subset_filtered.csv")
    trust = df["trstplt"]
    income = df["hinctnta"]

    [xj,nj,xk,nk] = generateVariables(trust,income, Threshold)

    thetaj = xj/nj
    thetak = xk/nk

    print('Uppgift A2\n')
    print(f'Andelen positiva svar i låginkomstgruppen är {round(thetaj,3)}')
    print(f'Andelen positiva svar i höginkomstgruppen är {round(thetak,3)}') 
    print(f'Skillnaden i andelarna är {round(thetaj-thetak,3)}\n\n')

    [decision, DeltaConj] = A3_b(xj,nj,xk,nk)
    print('Uppgift A3\n')
    print(f'Bayesbeslutet väljer Thetaj>Thetak: {decision}\n\n')

    [mu_j, mu_k, sigmaSquare] = A4_a(xj,xk,nj,nk) # very slow, data printed below

    ## Results of A4_a
    # mu_j = -1.6551
    # mu_k = -1.0011
    # sigmaSquare = 0.1

    Delta2 = förkastelseMetoden(xj, xk, nj, nk, mu_j, mu_k, sigmaSquare)

    print('Uppgift A4\n')
    print(f'Aposterioriväntevärde för Delta är {np.mean(Delta2)}')
    se = Delta2.std(ddof=1) / np.sqrt(len(Delta2)) ## std(ddof=1) ger s^2 = 1/(n-1) Sum (x_i - \bar x)^2
    print(f'Montecarlo-medelfelet för förkastelsemetoden är {se}')

    plt.title('Comparison between posteriors for conjugate and non-conjugate priors')
    plt.hist(DeltaConj, bins = 50, density = True, alpha = 0.6, label='Posterior with Conjugate prior')
    plt.hist(Delta2, bins = 50, density = True, alpha = 0.6, label='Posterior for logit-normal prior')
    plt.legend()
    plt.show()



def generateVariables(trust,income, Threshold):
    xj = len(income.loc[(trust >= Threshold) & (income <= 3)])
    nj = len(income.loc[income <= 3])
    xk = len(income.loc[(trust >= Threshold) & (income >= 8)])
    nk = len(income.loc[income >= 8])
    return [xj,nj,xk,nk]


def A3_b(xj,nj,xk,nk):
    Betaj = BetaDistribution(xj,nj-xj,10000).generateBetaDistribution()
    Betak = BetaDistribution(xk,nk-xk,10000).generateBetaDistribution()

    Delta = Betaj-Betak
    posDelta = Delta[Delta <= 0]

    bayesDecision = len(posDelta)  / len(Delta)

    # plt.hist(Delta, density=True, bins = 50)
    # plt.show()

    if (bayesDecision <0.5):
        return ['a1', Delta]
    elif (bayesDecision >0.5):
        return ['a0', Delta]
    else:
        return ['Expected loss is equal', Delta]


def A4_a(xj,xk,nj,nk): ## A4.a) calculation of parameters for prior (logit-normal)

    sigmaSquare = 0.1
    n = 100 ## megaslow be careful
    iter = 10000
    Theta = np.linspace(-10,10,iter)

    expectedGuessJ = 0.15
    expectedGuessK = 0.25

    for i in range(iter):
        Normal = NormalDistribution(Theta[i],sigmaSquare,n).generateNormalDistribution()
        logitNormal = 1/(1+np.exp(-Normal))
        mean = np.mean(logitNormal)
        # print(mean)
        if mean < expectedGuessJ+0.01 and mean > expectedGuessJ-0.01:
            mu_j = Theta[i]
        if mean < expectedGuessK+0.01 and mean > expectedGuessK -0.01:
            mu_k = Theta[i]

    Xj = NormalDistribution(mu_j,sigmaSquare, n).generateNormalDistribution()
    Xk = NormalDistribution(mu_k, sigmaSquare, n).generateNormalDistribution()

    Thetaj = 1/(1+np.exp(-Xj))
    Thetak = 1/(1+np.exp(-Xk))

    ## prior predictives

    rng = np.random.default_rng()

    PriorPredj = np.zeros(n)
    PriorPredk = np.zeros(n)


    for i in range(len(Thetaj)):

        binJ = BinDistribution(nj,Thetaj[i],n).generateBinDistribution()
        binK = BinDistribution(nk,Thetak[i],n).generateBinDistribution()

        PriorPredj[i] = rng.choice(binJ)
        PriorPredk[i] = rng.choice(binK)


    ## Plots of prior predictive vs observed data, use to determine prior parameters
    plt.figure()
    plt.hist(PriorPredj,bins = 25, label= "Prior Predictive for j")
    plt.axvline(xj, color = "green", label = "Observed xj")
    plt.legend()
    plt.figure()
    plt.hist(PriorPredk, bins = 25, label="Prior Predictive for k")
    plt.axvline(xk, color = "green", label="Observed xk")
    plt.legend()
    plt.show()

    return [mu_j, mu_k, sigmaSquare]


def förkastelseMetoden(xj, xk , nj, nk, mu_j, mu_k,sigmaSquare): ## A4.b) simulation of Posterior

    n = 10000

    Logmj = sp.stats.binom.logpmf(xj, nj, xj / nj)
    Logmk = sp.stats.binom.logpmf(xk, nk, xk / nk)

    PosteriorJ = []
    PosteriorK = []

    max_iter = 1000000
    iter = 0 

    while (len(PosteriorJ) < n) and (iter < max_iter):
        Xj = sp.stats.norm.rvs(loc=mu_j, scale=np.sqrt(sigmaSquare))
        Thetaj = sp.special.expit(Xj) 
        Uj = sp.stats.uniform.rvs()
        if (np.log(Uj)) < sp.stats.binom.logpmf(xj,nj,Thetaj) - Logmj:
            PosteriorJ.append(Thetaj)
        iter+= 1

    if (iter == max_iter):
        print('Rejection sampling did not converge for ThetaJ')

    iter = 0

    while (len(PosteriorK) < n) and (iter < max_iter):
        Xk = sp.stats.norm.rvs(loc=mu_k, scale=np.sqrt(sigmaSquare))
        Thetak = sp.special.expit(Xk)
        Uk = sp.stats.uniform.rvs()
        if (np.log(Uk)) < sp.stats.binom.logpmf(xk,nk,Thetak) - Logmk:
            PosteriorK.append(Thetak)
        iter += 1

    if (iter == max_iter):
            print('Rejection sampling did not converge for ThetaK')

    PosteriorJ = np.array(PosteriorJ)
    PosteriorK = np.array(PosteriorK)


    if (len(PosteriorK) != len(PosteriorK)):
        print('Lengths did not match')
        return
    else:
        return PosteriorJ - PosteriorK
    




main()