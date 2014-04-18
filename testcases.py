from filters import *
from utils import *

def test_filters():
    df = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}])
    par = {'a': 0,
           'b': 1
           }
    method = 'trn_dev_ks'

    filter(df,method,par)

def check_policies():
    train = readZipCSV("../data/", "train.csv.zip")
    train['ABCDEFG'] = train['A'].apply(str)+train['B'].apply(str)+train['C'].apply(str)+train['D'].apply(str)+train['E'].apply(str)+train['F'].apply(str)+train['G'].apply(str)
    customers = []
    d = {}
    for i in range(train.shape[0]):
        if train.iloc[i,0] != train.iloc[i-1,0]:
            d.clear()
        if train.iloc[i,2] == 0:
            if d.get(train.iloc[i,-1]) == None:
                d[train.iloc[i,-1]] = 1
            else:
                d[train.iloc[i,-1]] += 1
        else:
            print(i)
            if d.get(train.iloc[i,-1]) == None:
                customers.append(train.iloc[i,0])

    print("=========================================")
    print("Number of customers with unprepared purchase: %d" % len(customers))
    printList(customers,"%12d")
    print("=========================================")

def main():
    #test_filters()
    check_policies()

if __name__ == '__main__':
    main()
