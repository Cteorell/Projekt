import pandas as pd
from matplotlib import pyplot as plt
from Distributions import BetaDistribution


def main():
    N = 1566 ## replies
    Threshold = 6


    df = pd.read_csv("ESS11e04_2-subset_filtered.csv")
    trust = df["trstplt"]
    income = df["hinctnta"]

    [xj,nj,xk,nk] = generateVariables(trust,income, Threshold)

    thetaj = xj/nj
    thetak = xk/nk

    print(f'Andelen positiva svar i låginkomstgruppen är {round(thetaj,3)*100} %')
    print(f'Andelen positiva svar i höginkomstgruppen är {round(thetak,3)*100} %') 

    print(nj)
    print(nk)

    decision = A3_b(xj,nj,xk,nk)
    print(decision)


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
    posDelta = Delta[Delta >0]

    bayesDecision = len(posDelta)  / len(Delta)


    plt.hist(Delta, density=True, bins = 50)
    plt.show()

    if (bayesDecision <0.5):
        return 'a1'
    elif (bayesDecision >0.5):
        return 'a0'
    else:
        return 'Expected loss is equal'



main()