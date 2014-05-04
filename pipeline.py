import sys
from sklearn import preprocessing # add by sandy
from utils import *
# import configuration file and figure out what to run
from config import *
import random


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

def recode_features_2b(df, par):
    """
    Recode some continuous original features and append to the same data set
    df[0] is the dataset to be recoded (train/test).
    df[1] is the dataset that provides help (train).
    notice that this function is project specific -- Allstate only
    """
    recoded_features = df[0]
    helper_features = df[1]

    recoded_features['hour'] = recoded_features['time'].apply(lambda x: int(x.split(':')[0]))
    recoded_features['r_hour'] = pd.cut(recoded_features['hour'],[-1,6,12,18,24]).labels
    recoded_features['r_car_age'] = pd.cut(recoded_features['car_age'],[-1,3,7,12,100]).labels
    recoded_features['r_age_oldest'] = pd.cut(recoded_features['age_oldest'],[-1,28,44,60,100]).labels
    recoded_features['r_age_youngest'] = pd.cut(recoded_features['age_youngest'],[-1,26,40,57,100]).labels
    recoded_features['r_cost'] = pd.cut(recoded_features['cost'],[-1,605,635,665,1000]).labels
    bought = helper_features.loc[helper_features['record_type']==1,:]
    bought_count = bought.groupby('location')['location'].count()
    rhash = bought_count.apply(lambda x: int(x>10)+int(x>15)+int(x>25))
    recoded_features['r_location'] = recoded_features['location'].apply(lambda row: rhash.get(row))

    return [recoded_features]


def analyze_2fy(df, par):
    """
    notice that this function is project specific -- Allstate only
    """
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
    """
    Create static features based on original/recoded features and return a new data set
    df[0] is the dataset to create features on (train/test).
    df[1] is the dataset that provides help (train).
    notice that this function is project specific -- Allstate only
    """

    input_features = df[0]
    helper_features = df[1]

    firstVarList = par['condprob'][0]
    secondVarList = par['condprob'][1]

    if not firstVarList:
        print("ERROR: failed to create static features based on an empty list.")
        return None

    bought = helper_features.loc[helper_features['record_type']==1,:]
    static_features = {}

    if not secondVarList:
        for A in firstVarList:
            print("Generating cpt for {}".format(A))
            cpt = genCondProbTable(bought,[A],[])
            static_features['p_'+A] = genCondProbVar(input_features,cpt)
    else:
        for A in firstVarList:
            for B in secondVarList:
                print("Generating cpt for {} | {}".format(A,B))
                cpt = genCondProbTable(bought,[A],[B])
                static_features['p_'+A+'_'+B] = genCondProbVar(input_features,cpt)

    return [pd.DataFrame(static_features)]

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
#taku: feature selection
def feature_selection_4a(df, par):
    df_data = df.iloc[:,0:len(df.columns) - 1]
    df_y = df.iloc[:,len(df.columns) - 1]
    df_data_new = LinearSVC(C=par['C'], penalty=par['penalty'], dual=par['dual']).fit_transform(df_data, df_y)
    df_new = pd.concat([df_data_new, df_y], axis=1)
    return df_new

# This method splits 'train_ready' into train/development datasets and stores them
# in files
def split_data_5a(df, par):

    # read from a file
    # TODO: change it to directly read from df
    df = pd.read_csv(par['dir']+par['fname'])
    # find number of rows and columns
    num_col = len(df.columns)
    num_row = len(df.index)
    print(num_col, num_row)

    # read the train_ratio
    # it is the ratio of trainset to the whole set
    train_ratio= par['train_ratio']

    # read the random seed
    random.seed(par['seed'])

    # split the data
    # splitting is based on customer ID:
    # train_ratio (e.g., 70%) of customers will be in trainset
    new_customer_flag = True
    train_flag = False
    train_arr = np.zeros(num_row)
    for r in range(num_row):
        if new_customer_flag:
            train_flag = (random.random() < train_ratio)
            new_customer_flag = False
        if train_flag:
            train_arr[r] = 1
        # check the record type column to see whether it is
        # a new customer ID
        if df.iloc[r,num_col-1] == 1:
            new_customer_flag = True
    print(train_arr)

    # split the data into trainset and devset
    # for now, labels are in the last column of trainset and devset
    trainset = df.iloc[train_arr==1,:]
    devset = df.iloc[train_arr==0,:]

    return [trainset, devset]

# This method runs the SVM train model over the trainset.csv and
# evaluate its performance over the devset.csv
# input: trainset.csv, devset.csv, model parameters
# output: logging error rate per parameter combination
def model_train_dev_svm(df, par):

    # get the trainset and devset
    trainset = df[0]
    devset = df[1]

    # find number of rows and columns
    num_col = len(trainset.columns)
    num_row = len(trainset.index)

    # get the features and labels
    train_feature = trainset.iloc[:, 0:(num_col-1)].values
    train_label = trainset.iloc[:,num_col-1].values

    test_feature = devset.iloc[:, 0:(num_col-1)].values
    test_label = devset.iloc[:,num_col-1].values

    #print(train_feature, train_label, test_feature, test_label)
    print('train/test features and targets extracted')

    # start to train
    svm_model = svm_train(train_feature, train_label)
    svm_test(svm_model, test_feature, test_label, devset, False)

    return []

def main():
    # The following lines do not need tuning in most cases
    steps = {'0a': test_only_0a,
            '1a': read_train_data_1a,
            '1b': read_test_data_1b,
            '2a': summarize_data_2a,
            '2b': recode_features_2b,
            '2fy': analyze_2fy,
            '2sandy': preprocess_data_2sandy,
            '3a': create_static_features_3a,
            '4a': feature_selection_4a,
            '5a': split_data_5a,
            #'1sandy': get_train_purchase_data_1sandy
            '6a': model_train_dev_svm
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
        inputDataframes = lookupDict(datasets,df_in[id])
        if isinstance(df_out[id],list):
            # if we need to output multiple datasets, then the 'steps' function must return a list of dataframe
            retDataframes = steps[id](inputDataframes,pars[id])
            for dfName, df in zip(df_out[id],retDataframes):
                datasets[dfName] = df
        else:
            datasets[df_out[id]] = steps[id](inputDataframes,pars[id])
        # write dataframes to .csv files on disk as needed
        if id in df_to_write:
            if isinstance(df_to_write[id],list):
                for dfName in df_to_write[id]:
                    datasets[dfName].to_csv('data/'+dfName+'.csv',index=False)
            else:
                datasets[df_to_write[id]].to_csv('data/'+df_to_write[id]+'.csv',index=False)

if __name__ == "__main__":
    main()
