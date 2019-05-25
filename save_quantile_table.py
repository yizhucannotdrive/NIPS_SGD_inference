import pickle
quantile_table = dict()
splittings = ["C", "E"]
ms = [10,20,30,40,60,100,120,150]
ds = [1,2,3,4,10,50,80,100]
for splitting in splittings:
    quantile_table[splitting] =dict()
    for m in ms:
        quantile_table[splitting][m] = dict()
quantile_table["C"][10][1] = 2.93
quantile_table["C"][10][2] = 6.58
quantile_table["C"][10][3] = 12.09
quantile_table["C"][10][4] = 20.98
quantile_table["C"][20][1] = 2.18
quantile_table["C"][20][2] = 4.22
quantile_table["C"][20][3] = 6.55
quantile_table["C"][20][4] = 9.35
quantile_table["C"][30][1] = 1.91
quantile_table["C"][30][2] = 3.54
quantile_table["C"][30][3] = 5.28
quantile_table["C"][30][4] = 7.22
quantile_table["C"][40][1] = 1.76
quantile_table["C"][40][2] = 3.18
quantile_table["C"][40][3] = 4.66
quantile_table["C"][40][4] = 6.28
quantile_table["C"][30][10] = 25.18
pickle.dump(quantile_table, open("quantile_table", "wb"))
