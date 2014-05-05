#exec_seq = ['1a','3a']
exec_seq = ['6a']

pars = {'0a': None,
        '1a': {'dir': "../data/", 'fname': "train.csv.zip"},
        '1b': {'dir': "../data/", 'fname': "test.csv.zip"},
        '2a': None,
        '2b': None,
        '2fy': {'log': "../log/step_2fy.log"},
        '2sandy': {'missing': '4'},
        '3a': {'condprob':(('A','B','C','D','E','F'),('day','state','group_size','homeowner','car_value','risk_factor','married_couple','C_previous','duration_previous'))},
        #'3a': {'condprob':(('A','B'),('day','state'))} # for debugging only
        '4a': {'C': '0.01', 'penalty': 'l1', 'dual': 'False'},
        '5a': {'train_ratio': 0.7, 'seed': '1000', 'dir': 'data/', 'fname': 'train_5sandy_all.csv'},
        '6a': None
        }

df_in = {'0a': ['test_only_in1','test_only_in2'],
        '1a': None,
        '1b': None,
        '2a': 'train',
        '2b': ['train','train'], # the latter should be train at most times
        '2fy': 'train',
        '2sandy': 'train_3a',
        '3a': ['train','train'],  # the latter should be train at most times
		'4a': 'train_3a',
        '5a': ['train_ready', 'train_target'],
        '6a': ['trainset', 'devset']
        }

df_out = {'0a': ['test_only_out1','test_only_out2'],
        '1a': 'train',
        '1b': 'test',
        '2a': None,
        '2b': ['train'],
        '2fy': None,
        '2sandy': 'train_2sandy',
        '3a': ['static'],
        '4a': 'train_4a',
        '5a': ['trainset', 'devset'],
        '6a': []
        }

# Names of datasets to be read from disk
# Can be a string or a list of strings
df_to_read = {'0a': ['test_only_in1','test_only_in2'],
              '5a': ['train_ready', 'train_target'],
              '6a': ['trainset', 'devset']
                }

# Names of datasets to be written to disk
# Can be a string or a list of strings
df_to_write = {'0a': ['test_only_out1','test_only_out2'],
               '2b': 'train',
               '3a': 'static',
               '5a': ['trainset', 'devset']
                }
