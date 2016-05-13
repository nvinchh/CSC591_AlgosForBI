import sys
import csv
import math
import random

# Finding advertisers matching the keyword
def fAdv(query, bidders):
    advertisers = [i for i in bidders if i[1] == query]
    return advertisers

# Finding advertisers which have budget available in order to bid
def fAdvWBudget(advertisers, budget):
    advWBudget = []
    for adv in advertisers:
        if budget[adv[0]] >= float(adv[2]):
            advWBudget.append(adv)
    return advWBudget

# Greedy Algorithm
def GreedyAlgo(advWBudget):
    highest = max(adv[2] for adv in advWBudget)
    select = [adv for adv in advWBudget if adv[2] == highest]
    return select[0]

# MSSV Algo
def MSSVAlgo(advWBudget, budget, initBudget):
    max = 0.0
    select = []
    for adv in advWBudget:
        xu = (initBudget[adv[0]] - budget[adv[0]])/initBudget[adv[0]]
        fXu = 1 - math.exp(xu-1)
        if (float(adv[2]) * fXu) > max:
            max = (float(adv[2]) * fXu)
            select = adv
    return select

# Balance Algo
def BalanceAlgo(advWBudget, budget):
    max = 0
    for adv in advWBudget:
        if budget[adv[0]] > max:
            max = budget[adv[0]]
            select = adv
    return select

def main(argv):
    ch = str(argv)
    #print ch
    cList = ["greedy","msvv","balance"]

    if ch not in cList:
        print "Please type the correct choice"
    else:
        # Loading the 'Queries' and 'Bidders' Datasets from the files
        queries = []
        fQueries = open("queries.txt")
        try:
            for query in fQueries:
                queries.append(query.replace("\n",""))
        finally:
            fQueries.close()

        bidders = []
        fBidder = open("bidder_dataset.csv")
        try:
            rows = csv.reader(fBidder)
            for row in rows:
                bidders.append(row)
        finally:
            fBidder.close()


        # Storing the mapping of advertisers with budgets
        bDict = {}
        for bidder in bidders[1:]:
            if(bidder[3]):
                bDict[bidder[0]] = float(bidder[3])
        initBudget = bDict.copy()

        optSum = sum(bDict.values())

        #Finding the revenue using the algorithm provided in argument
        totalrevenue = 0.0
        random.seed(0)
        for i in range(0, 100):
            revenue = 0.0
            random.shuffle(queries)
            bDict = initBudget.copy()
            for query in queries:
                advertisers = fAdv(query, bidders)
                advWBudget = fAdvWBudget(advertisers,bDict)
                if len(advWBudget) >= 1:
                    if ch == "greedy":
                        chosenBidder = GreedyAlgo(advWBudget)
                    elif ch == "msvv":
                        chosenBidder = MSSVAlgo(advWBudget, bDict, initBudget)
                    else:
                        chosenBidder = BalanceAlgo(advWBudget,bDict)
                    revenue = revenue + float(chosenBidder[2])
                    bDict[chosenBidder[0]] = bDict[chosenBidder[0]] - float(chosenBidder[2])
            totalrevenue = totalrevenue + revenue
        avgRevenue = totalrevenue/100

        print "Revenue: "+ str(avgRevenue)
        print "Competitive Ratio: "+ str(avgRevenue/optSum)


if __name__ == "__main__":
    if (len(sys.argv) == 2):
        main(sys.argv[1])
    else:
        print "Invalid number of arguments"















