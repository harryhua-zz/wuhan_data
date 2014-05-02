#exec_seq = ['1a','3a']
exec_seq = ['0a']

pars = {'0a': None,
        '1a': {'dir': "../data/", 'fname': "train.csv.zip"},
        '1b': {'dir': "../data/", 'fname': "test.csv.zip"},
        '2a': None,
        '2fy': {'log': "../log/step_2fy.log"},
        '2sandy': {'missing': '4'},
        '3a': {'condprob':(('A','B','C','D','E','F'),('day','state','group_size','homeowner','car_value','risk_factor','married_couple','C_previous','duration_previous'))}
        #'3a': {'condprob':(('A','B'),('day','state'))} # for debugging only
        '4a': {'C': '0.01', 'penalty': 'l1', 'dual': 'False'}
        }

df_in = {'0a': ['test_only_in1','test_only_in2'],
        '1a': None,
        '1b': None,
        '2a': 'train',
        '2fy': 'train',
        '2sandy': 'train_3a',
        '3a': 'train',
        '4a': 'train_3a'
        }

df_out = {'0a': ['test_only_out1','test_only_out2'],
        '1a': 'train',
        '1b': 'test',
        '2a': None,
        '2fy': None,
        '2sandy': 'train_2sandy',
        '3a': 'train_3a',
        '4a': 'train_4a'
        }

# Names of datasets to be read from disk
# Can be a string or a list of strings
df_to_read = {'0a': ['test_only_in1','test_only_in2']
                }

# Names of datasets to be written to disk
# Can be a string or a list of strings
df_to_write = {'0a': ['test_only_out1','test_only_out2'],
                '3a': 'train_3a'
                }