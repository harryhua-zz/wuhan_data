nametag = 'test'
isTest = True
exec_seq = ['1b','2a','2b','3a','3b','3z','4a']

pars = {'0a': None,
        '1a': {'dir': "../data/", 'fname': "train.csv.zip"},
        '1b': {'dir': "../data/", 'fname': "test_v2.csv.zip"},
        '1z': {'train_ratio': 0.7, 'seed': '1000', 'dir': 'data/', 'fname': 'train_5sandy_test_100.csv'},
        '2a': {'columns':['state','location','group_size','homeowner','car_age','car_value','risk_factor',\
                'age_oldest','age_youngest','married_couple','C_previous','duration_previous']},
        '2b': None,
        '2fy1': {'log': "../log/step_2fy.log"},
        '2fy2': None,
        '3a': {'condprob':(('A','B','C','D','E','F'),\
                ('day','state','group_size','homeowner','car_value',\
                'risk_factor','married_couple','C_previous','duration_previous','r_hour','r_location',\
                'r_car_age','r_age_oldest','r_age_youngest','r_cost'))},
        #'3a': {'condprob':(('A','B'),('day','r_hour'))}, # for debugging only
        '3b': {'method_duplicate': 0}, # 0 presents original, 1 presents only handle record_type==0
        #'3z': 0, # 0 presents for train, 1 presents for test
        '3z': {'handle_duplicate': 1}, # 1 presents do, 0 presents do not
        '4a': {'missing': '4'},
        '4b': {'C' : 0.01, 'penalty' : 'l1', 'dual' : False, 'isTest': isTest},
        '6a': {'model': 'random_forest'},
        '6b': {'mode': 'dev', 'model' : 'random_forest'}
        }

df_in = {'0a': 'test_only_in1',
        '1a': None,
        '1b': None,
        '1z': 'train',
        '2a': nametag,
        '2b': [nametag,'trainset'], # the latter should be trainset at most times
        '2fy1': 'train',
        '2fy2': 'train',
        '3a': [nametag,'trainset'],  # the latter should be trainset at most times
        '3b': [nametag],
        '3z': [nametag,nametag+'_static',nametag+'_dynamic'],
        '4a': nametag+'_pool',
        '4b': [nametag+'_preprocessing','selected_features',nametag+'_target'],
        '6a': ['trainset_ready','trainset_target'],
        '6b': ['devset0_ready', 'devset0_customer_ID', 'devset1']
        }

df_out = {'0a': ['test_only_out1','test_only_out2'],
        '1a': 'train',
        '1b': 'test',
        '1z': ['trainset', 'devset0', 'devset1'],
        '2a': nametag,
        '2b': [nametag],
        '2fy1': None,
        '2fy2': None,
        '3a': [nametag+'_static'],
        '3b': nametag+'_dynamic',
        '3z': [nametag+'_pool',nametag+'_target',nametag+'_customer_ID'],
        '4a': nametag+'_preprocessing',
        '4b': [nametag+'_ready','selected_features'],
        '6a': [],
        '6b': ['predict_options']
        }

# Names of datasets to be read from disk
# Can be a string or a list of strings
df_to_read = {'0a': 'test_only_in1',
              '1b': 'trainset',
              #'1z': 'train',
              #'2a': [nametag,'trainset'],
              #'2b': [nametag,'trainset'], # the latter should be trainset at most times
              #'3a': [nametag,'trainset'],  # the latter should be trainset at most times
              #'3b': [nametag,'devset0_static','selected_features'],
              #'3z': [nametag,nametag+'_static',nametag+'_dynamic'],
              #'4a': nametag+'_pool',
              '4b': [nametag+'_preprocessing','selected_features',nametag+'_target'],
              '6a': ['trainset_ready','trainset_target'],
              '6b': ['devset0_ready', 'devset0_customer_ID', 'devset1']
            }

# Names of datasets to be written to disk
# Can be a string or a list of strings
df_to_write = {'0a': ['test_only_out1','test_only_out2'],
        '1z': ['trainset', 'devset0', 'devset1'],
        '2b': [nametag],
        '3a': [nametag+'_static'],
        '3b': nametag+'_dynamic',
        '3z': [nametag+'_pool',nametag+'_target',nametag+'_customer_ID'],
        '4a': nametag+'_preprocessing',
        '4b': [nametag+'_ready','selected_features'],
        '6b': ['predict_options']
        }

