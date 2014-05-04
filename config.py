#exec_seq = ['1a','3a']
exec_seq = ['1a','2a','2b','3a']

pars = {'0a': None,
        '1a': {'dir': "../data/", 'fname': "train.csv.zip"},
        '1b': {'dir': "../data/", 'fname': "test.csv.zip"},
        '2a': None,
        '2b': None,
        '2fy1': {'log': "../log/step_2fy.log"},
        '2fy2': None,
        '2sandy': {'missing': '4'},
        '3a': {'condprob':(('A','B','C','D','E','F'),('day','state','group_size','homeowner','car_value','risk_factor','married_couple','C_previous','duration_previous','r_hour','r_location','r_car_age','r_age_oldest','r_age_youngest','r_cost'))},
        #'3a': {'condprob':(('A','B'),('day','r_hour'))}, # for debugging only
        '4a': {'C': '0.01', 'penalty': 'l1', 'dual': 'False'},
        '5a': {'train_ratio': 0.7, 'seed': '1000', 'dir': 'data/', 'fname': 'train_5sandy_test_100.csv'},
        '6a': None
        }

df_in = {'0a': 'test_only_in1',
        '1a': None,
        '1b': None,
        '2a': 'train',
        '2b': ['train','train'], # the latter should be train at most times
        '2fy1': 'train',
        '2fy2': 'train',
        '2sandy': 'train_3a',
        '3a': ['train','train'],  # the latter should be train at most times
		'4a': 'train_3a',
        '5a': [],
        '6a': ['trainset', 'devset']
        }

df_out = {'0a': ['test_only_out1','test_only_out2'],
        '1a': 'train',
        '1b': 'test',
        '2a': 'train',
        '2b': ['train'],
        '2fy1': None,
        '2fy2': None,
        '2sandy': 'train_2sandy',
        '3a': ['static'],
        '4a': 'train_4a',
        '5a': ['trainset', 'devset'],
        '6a': []
        }

# Names of datasets to be read from disk
# Can be a string or a list of strings
df_to_read = {'0a': ['test_only_in1','test_only_in2'],
              '6a': ['trainset', 'devset']
                }

# Names of datasets to be written to disk
# Can be a string or a list of strings
df_to_write = {'0a': ['test_only_out1','test_only_out2'],
               '2a': 'train',
               '2b': 'train',
               '3a': 'static',
               '5a': ['trainset', 'devset']
                }
