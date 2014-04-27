import sys
from sklearn import preprocessing # add by sandy
from utils import *
# import configuration file and figure out what to run
from config import *

def test_only_0a(df,par):
# testing multiple datasets I/O and .csv I/O
    print(df[0])
    print(df[1])
    return df

def read_train_data_1a(df, par):
    return readZipCSV(par['dir'], par['fname'])

def read_test_data_1b(df, par):
    return readZipCSV(par['dir'], par['fname'])

def summarize_data_2a(df, par):
    colNames = list(df.columns.values)
    print("==================================================== Summary ===================================================")
    print("nrows: ",df.shape[0],"ncols: ",df.shape[1])
    print(df.describe())
    print("%20s %15s %15s" % ("column","num_missing","missing_rate"))
    for c in colNames:
        nMiss = df[c].isnull().sum()
        print("%20s %15d %15.4f" % (c,nMiss,nMiss/df.shape[0]))
    print("================================================================================================================")
    return None

def analyze_2fy(df, par):
    if par.get('log') != None:
        log = open(par['log'], 'w')
    else:
        log = sys.stdout
    df['hour'] = df.loc[:,'time'].apply(lambda x: x.split(':')[0])

    for c in ['state','location','day','hour']:
        print1DRiskTable(gen1DRiskTable(df,c,'record_type',1),log)

    if par.get('log') != None:
        log.close()
    return None

def create_static_features_3a(df, par):
# TODO: This function fails when par['condprob'][0] or par['condprob'][1] has only one variable in the list
    firstVarList = par['condprob'][0]
    secondVarList = par['condprob'][1]

    if not firstVarList:
        print("ERROR: failed to create static features based on an empty list.")
        return None

    bought = df.loc[df['record_type']==1,:]
    static_features = {}

    if not secondVarList:
        for A in firstVarList:
            print("Generating cpt for {}".format(A))
            cpt = genCondProbTable(bought,[A],[])
            #print(cpt)
            static_features['p_'+A] = genCondProbVar(df,cpt)
    else:
        for A in firstVarList:
            for B in secondVarList:
                cpt = genCondProbTable(bought,[A],[B])
                static_features['p_'+A+'_'+B] = genCondProbVar(df,cpt)

    return pd.DataFrame(static_features)

#def create_dynamic_features_3b(df, par):

#sandy: data preprocessing
def preprocess_data_2sandy(df, par):
    print(par['missing'])
    #sandy: handle the missing value
    filtered_train = pd.DataFrame([])
    if cmp(par['missing'], '1') == 0:
        print('par_missing == 1')
        filtered_train = df.dropna(axis=0) # drop rows with NA
    elif cmp(par['missing'], '2') == 0:
        filtered_train = df.dropna(axis=1) # drop columns with NA
    elif cmp(par['missing'], '3') == 0:
        filtered_train = df.fillna(value = 0) # fill missing value with 0 (this execution is not work now)
    elif cmp(par['missing'], '4') == 0:
        filtered_train = df.fillna(df.mean())     # fill missing value with mean

    """
    filtered_train_scale = preprocessing.scale(filtered_train.to_records())
    filtered_train_scale_df = pd.DataFrame( filtered_train_scale[1:,1:],
                                           index=filtered_train_scale[1:,0], columns=filtered_train_scale[0,1:] )
    """
    #sandy: normalize the data
    filtered_train_norm = (filtered_train - filtered_train.mean()) / (filtered_train.max() - filtered_train.min())

    return filtered_train_norm

"""
#sandy: only keep the purchase record in train
def get_train_purchase_data_1sandy(df, par):
    df_purchase = df[df['record_type'] == 1]
    #print df_purchase
    return df_purchase
"""


def main():
    # The following lines do not need tuning in most cases
    steps = {'0a': test_only_0a,
            '1a': read_train_data_1a,
            '1b': read_test_data_1b,
            '2a': summarize_data_2a,
            '2fy': analyze_2fy,
            '2sandy': preprocess_data_2sandy,
            '3a': create_static_features_3a
            #'1sandy': get_train_purchase_data_1sandy
            }

    datasets = {None: None}

    for id in exec_seq:
        # read in dataframes from .csv files on disk as needed
        if id in df_to_read:
            if isinstance(df_to_read[id],list):
                for dfName in df_to_read[id]:
                    datasets[dfName] = pd.read_csv('data/'+dfName+'.csv')
            else:
                datasets[df_to_read[id]] = pd.read_csv('data/'+df_to_read[id]+'.csv')
        # major loop that calls all the steps in execution sequence
        if isinstance(df_out[id],list):
            # if we need to output multiple datasets, then the 'steps' function must return a list of dataframe
            inList = []
            for dfName in df_in[id]:
                inList.append(datasets[dfName])
            retList = steps[id](inList, pars[id])
            for dfName, df in zip(df_out[id],retList):
                datasets[dfName] = df
        else:
            datasets[df_out[id]] = steps[id](datasets[df_in[id]], pars[id])
        # write dataframes to .csv files on disk as needed
        if id in df_to_write:
            if isinstance(df_to_write[id],list):
                for dfName in df_to_write[id]:
                    datasets[dfName].to_csv('data/'+dfName+'.csv',index=False)
            else:
                datasets[df_to_write[id]].to_csv('data/'+df_to_write[id]+'.csv',index=False)

if __name__ == "__main__":
    main()
