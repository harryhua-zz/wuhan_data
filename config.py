exec_seq = ['1a','3a']

pars = {'1a': {'dir': "../data/", 'fname': "train.csv.zip"},
        '1b': {'dir': "../data/", 'fname': "test.csv.zip"},
        '2a': None,
        '2fy': {'log': "../log/step_2fy.log"},
        '2sandy': {'missing': '4'},
        '3a': {'condprob':(('A','B','C','D','E','F'),('day','state','group_size','homeowner','car_value','risk_factor','married_couple','C_previous','duration_previous'))}
        #'3a': {'condprob':(('A','B'),('day','state'))} # for debugging only
        }

df_in = {'1a': None,
        '1b': None,
        '2a': 'train',
        '2fy': 'train',
        '2sandy': 'train_3a',
        '3a': 'train'
        }

df_out = {'1a': 'train',
        '1b': 'test',
        '2a': None,
        '2fy': None,
        '2sandy': 'train_2sandy',
        '3a': 'train_3a'
        }


