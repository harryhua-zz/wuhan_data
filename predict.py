import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from datetime import datetime
import os, sys, warnings
import re

warnings.simplefilter(action='ignore', category=DeprecationWarning)

DEBUG=False

NOPT=7
MAX_OBS = 10003 #942

def plan_to_string(vec):
    """
    Assume plan is passed as a vector of length NOPT
    """
    out = ''
    for opt in vec:
        out += str(int(opt))
    return out

def accfun(act, pred):
    return sum(act==pred)/(1.0*len(act))

def aonfun(act, pred):
    return sum(np.all(act==pred,axis=1))/(1.0*len(act))

def moving_average(df,r,c,first,ma_size=3,slide_times=3):
    results = []
    for t in range(slide_times):
        lb = r-ma_size+1-t
        lb = lb if lb > first else first
        ub = r-t
        ub = ub if ub > lb else lb
        results.append(df.iloc[lb:(ub+1),c].mean())
    return results

def count_changes(df,r,c,first):
    results = [0,0,0]
    for i in range(first,r):
        results[0] += 1 if df.iat[i,c] != df.iat[i+1,c] else 0
        results[1] += df.iat[i+1,c] - df.iat[i,c]
        results[2] += abs(df.iat[i+1,c] - df.iat[i,c])
    return results

def genCondProbTable(df, listA, listB):
    if not listA:
        print("ERROR: Can't calculate probabilities for an empty list of variables.")
        return None

    totals = df.groupby(listA+listB)[listA[0]].count()

    if listB:
        return totals.groupby(level=listB).apply(lambda s: s.astype(float)/s.sum())
    else:
        return totals/totals.sum()

def genCondProbVar(df,cpt,cols):
    # TODO: cannot deal with missing values yet
    out = []
    if len(cols) == 1:
        df.loc[:,cols[0]].apply(lambda row: out.append(cpt.xs(row,level=cols[0],drop_level=False).values[:-1]))
        return np.array(out)
    elif len(cols) >= 2:
        df.loc[:,cols].apply(lambda row: out.append(cpt.xs(row.values,level=cols,drop_level=False).values[:-1]))
        return np.array(out)
    else:
        print("ERROR: Empty conditional probability hash table.")
        return None

def makeCondProbTableName(A, B):
    tname = 'p_'+A[0]
    for var in A[1:]:
        tname += '-'+var
    tname += '_'+B[0]
    for var in B[1:]:
        tname += '-'+var
    return tname

def create_cond_prob_tables(df, listA, listB):
    """
    Both listA and listB should be a list of lists of variables, e.g. listA = [['A','B'],['C'],['D','E','F']], listB = [['car_value','car_age']]
    """
    if not listA:
        print("ERROR: Can't calculate probabilities for an empty list of variables.")
        return None

    results = {}
    for A in listA:
        for B in listB:
            name = makeCondProbTableName(A,B)
            results[name] = genCondProbTable(df,A,B)
    return results

def read_cond_prob_tables(dirName):
    results = {}
    for path, subdirs, files in os.walk(dirName):
        for fileName in files:
            if fileName.endswith('.p'):
                name = os.path.splitext(fileName)[0]
                results[name] = pd.read_pickle(os.path.join(path,fileName))
    return results

def check_cond_prob_tables(dirName):
    count = 0
    for path, subdirs, files in os.walk(dirName):
        for fileName in files:
            if fileName.endswith('.p'):
                count += 1
    return count

def create_static_features(df,cpts,targetNamePattern):
    first_cpt_flag = True
    for name, cpt in sorted(cpts.items(), key=lambda t: t[0]):
        # find the column names that are conditioned on
        print(name)
        cols = [x for x in cpt.index.names if not re.search(targetNamePattern, x)]
        if first_cpt_flag:
            static = genCondProbVar(df,cpt,cols)
            first_cpt_flag = False
        else:
            var = genCondProbVar(df,cpt,cols)
            static = np.concatenate((static,var),axis=1)
    return static

def recode_features(df):
    car_value_map = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8}
    df['car_value'] = df['car_value'].apply(lambda x: car_value_map.get(x))
    df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
    df['r_hour'] = pd.cut(df['hour'],[-1,6,12,18,24]).labels
    df['r_car_age'] = pd.cut(df['car_age'],[-1,3,7,12,100]).labels
    df['r_age_oldest'] = pd.cut(df['age_oldest'],[-1,28,44,60,100]).labels
    df['r_age_youngest'] = pd.cut(df['age_youngest'],[-1,26,40,57,100]).labels
    df['r_cost'] = pd.cut(df['cost'],[-1,605,635,665,1000]).labels

    if not DEBUG:
        df['location'] = df['location'].fillna(99999).astype(int)
        # Assume each customer just has one row now
        location_count = df.groupby('location')['location'].count()
        location_hash = pd.Series(pd.qcut(location_count,4).labels, index=location_count.index)
        df['r_location'] = df['location'].apply(lambda x: location_hash.get(x))

    return df

def prepare_data(mode='train', n_to_last=3, gen_cpts=False):

    print("Entering prepare_data at {}".format(datetime.now()))
    if mode == 'train':
        input_data = pd.read_csv('../data/train.csv')
    elif mode == 'test':
        input_data = pd.read_csv('../data/test_v2.csv')
        n_to_last = 0
        gen_cpts = False
    else:
        print("ERROR: unrecognized mode. Exit.")
        sys.exit(1)

    if DEBUG:
        df = input_data.iloc[:MAX_OBS,:]
    else:
        df = input_data

    nrow = len(df)
    ncust = len(df.groupby('customer_ID'))

    pos_record_type = df.columns.values.tolist().index('record_type')
    pos_customer_ID = df.columns.values.tolist().index('customer_ID')
    pos_A = df.columns.values.tolist().index('A')
    pos_A_G = list(range(pos_A, pos_A+NOPT))
    pos_cost = df.columns.values.tolist().index('cost')
    pos_car_value = df.columns.values.tolist().index('car_value')

    plans_bought = np.zeros((ncust,NOPT))
    keep_index = []

    ma_size = 3
    n_ma_slots = 3
    ma_features = np.zeros((ncust,n_ma_slots*NOPT))
    cost_ma_features = np.zeros((ncust,n_ma_slots))

    n_chg_slots = 3
    chg_features = np.zeros((ncust,n_chg_slots*NOPT))
    cost_chg_features = np.zeros((ncust,n_chg_slots))

    first_opt_features = np.zeros((ncust,NOPT))

    print("Data setup is done at {}".format(datetime.now()))

    # Generate dynamic features and indices of rows to keep
    c = -1
    new_cust_flag = True
    first_index = 0
    curr_customer_ID = 0
    for r in range(nrow):
        if new_cust_flag:
            first_index = r
            curr_customer_ID = df.iat[r,pos_customer_ID]
            c += 1
            new_cust_flag = False
        if (r+1 < nrow and df.iat[r+1,pos_customer_ID] != curr_customer_ID) or r+1 == nrow:
            new_cust_flag = True
            output_index = r-n_to_last if r-n_to_last > first_index else first_index
            keep_index.append(output_index)
            for i,p in enumerate(pos_A_G):
                plans_bought[c,i] = df.iat[r,p]
                first_opt_features[c,i] = df.iat[first_index,p]
                ma_features[c,(i*n_ma_slots):((i+1)*n_ma_slots)] = moving_average(df,output_index,p,first_index,ma_size,n_ma_slots)
                chg_features[c,(i*n_chg_slots):((i+1)*n_chg_slots)] = count_changes(df,output_index,p,first_index)
            cost_ma_features[c,:n_ma_slots] = moving_average(df,output_index,pos_cost,first_index,ma_size,n_ma_slots)
            cost_chg_features[c,:n_chg_slots] = count_changes(df,output_index,pos_cost,first_index)

    print("Dynamic features creation is done at {}".format(datetime.now()))

    # Downsample original data -- each customer has one row only
    df_ind = df.iloc[keep_index,:].copy(deep=True)
    # Some necessary recoding before imputation and normalization
    df_ind = recode_features(df_ind)
    print("Recoding is done at {}".format(datetime.now()))
    keep_col = ['shopping_pt','day','group_size','homeowner','car_age','car_value','risk_factor',\
            'age_oldest','age_youngest','married_couple','C_previous','duration_previous',\
            'A','B','C','D','E','F','G','cost','hour','r_hour','r_car_age','r_age_oldest','r_age_youngest','r_cost']
    if not DEBUG:
        keep_col.append('r_location')
    df_ind_col = df_ind.loc[:,keep_col]

    # Generate static features
    df_ind_col_plans_bought = pd.DataFrame(np.concatenate((df_ind_col.values,plans_bought),axis=1),\
            columns=keep_col+['t_A','t_B','t_C','t_D','t_E','t_F','t_G'])

    if gen_cpts:
        cpts_out = create_cond_prob_tables(df_ind_col_plans_bought,\
                [['t_A'],['t_B'],['t_C'],['t_D'],['t_E'],['t_F'],['t_G']],\
                [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['r_cost']])
        count = check_cond_prob_tables('../data/')
        if count:
            print('Warning: {} conditional probability tables are found. '.format(count))
            print('         Newly generated cpts will not be written to disk.')
            print('         Remove them manually if you desire to write new cpts:')
            print('         rm ../data/*.p')
        else:
            for name, cpt in cpts_out.items():
                cpt.to_pickle('../data/'+name+'.p')

    cpts = read_cond_prob_tables('../data/')

    static_features = create_static_features(df_ind_col_plans_bought,cpts,'^t_')
    print("CPTs and static features are created at {}".format(datetime.now()))

    # Merge all the features -- original, dynamic and static
    df_ready = pd.DataFrame(np.concatenate((df_ind_col.values,first_opt_features,\
            ma_features,cost_ma_features,chg_features,cost_chg_features,static_features),axis=1))

    # Normalization
    print(df_ready.head())
    print("df_ready.shape={}".format(df_ready.shape))
    # Impute original/recoded data
    df_ready_mean = df_ready.mean()
    df_ready = df_ready.fillna(df_ready_mean)
    df_ready_norm = (df_ready-df_ready_mean)/(df_ready.max()-df_ready.min())
    print(df_ready_norm.head())
    print("Data preprocessing is done at {}".format(datetime.now()))

    return df_ready_norm.values, plans_bought, pd.DataFrame(df_ind['customer_ID'])

def train_cv(train, target_all):

    print("Entering train_cv at {}".format(datetime.now()))
    print("len(train)={}, len(target_all)={}".format(len(train),len(target_all)))
    # K-Fold cross validation
    cv = cross_validation.KFold(len(train), n_folds=5, indices=False)
    cfr = RandomForestClassifier(n_estimators=100, criterion='entropy')

    results = []
    for traincv, testcv in cv:
        pred = np.zeros((sum(testcv),NOPT))
        for i in range(NOPT):
            target = target_all[:,i]
            pred[:,i] = cfr.fit(train[traincv],target[traincv]).predict(train[testcv])
        results.append(aonfun(target_all[testcv,:], pred))

    print("Results: mean accuracy = {}".format(np.array(results).mean()))
    f = open("../data/results.log",'w')
    print("Results: mean accuracy = {}".format(np.array(results).mean()), file=f)
    f.close()
    print("Cross validation is done at {}".format(datetime.now()))

def train_predict(train, target_all, test, test_customer_ID):
    print("Entering train_predict at {}".format(datetime.now()))
    cfr = RandomForestClassifier(n_estimators=100, criterion='entropy')
    pred = np.zeros((len(test),NOPT))
    for i in range(NOPT):
        target = target_all[:,i]
        pred[:,i] = cfr.fit(train,target).predict(test)
    plans = np.apply_along_axis(plan_to_string,1,pred)
    return pd.DataFrame(np.concatenate((test_customer_ID.values,plans.reshape((len(pred),1))),axis=1),columns=['customer_ID','plan'])

def main():

    run = sys.argv[1]
    n_to_last_record = int(sys.argv[2])

    if run == 'prepare':
        train, target_all, junk_customer_ID = prepare_data('train', n_to_last_record, True)
        pd.DataFrame(train).to_csv('../data/train_'+str(n_to_last_record)+'.csv',index=False,float_format="%.4f")
        pd.DataFrame(target_all).to_csv('../data/target_all.csv',index=False,float_format="%.0f")
    elif run == 'train':
        train = pd.read_csv('../data/train_'+str(n_to_last_record)+'.csv')
        target_all = pd.read_csv('../data/target_all.csv')
        train_cv(train.values, target_all.values)
    elif run == 'test':
        test, junk_target, test_customer_ID = prepare_data('test', 0, False)
        pd.DataFrame(test).to_csv('../data/test_'+str(n_to_last_record)+'.csv',index=False,float_format="%.4f")
        pd.DataFrame(test_customer_ID).to_csv('../data/test_customer_ID'+str(n_to_last_record)+'.csv',index=False,float_format="%.0f")
        train = pd.read_csv('../data/train_'+str(n_to_last_record)+'.csv')
        target_all = pd.read_csv('../data/target_all.csv')
        submit = train_predict(train.values, target_all.values, test, test_customer_ID)
        submit.to_csv('../data/submission.csv',index=False)
    else:
        print("I don't know what to run...")
        sys.exit(2)

if __name__ == '__main__':
    main()

