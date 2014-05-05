step 1: 
    a: read original data set train.csv                             -- output datasets['train']
    b: read original data set test.csv                              -- output datasets['test']
step 2:
    a: filter customers with changed characteristics                -- input datasets['train'/'test'], output datasets['train']
    b: discretize/recode some original features                     -- input datasets['train'/'test'], output datasets['train']
step 3:
    a: create static features                                       -- input datasets['train'/'test'], output datasets['static'], to_csv('static.csv')
    b: create dynamic features                                      -- input datasets['train'/'test'], output datasets['dynamic'], to_csv('dynamic.csv')
    z: merge datasets, discard columns, split features/target       -- input datasets['train'/'test','static','dynamic'], output datasets['train_pool','train_target'], to_csv('train_pool.csv','train_target.csv') 
step 4:
    a: preprocess train/test pool, handle missing, normalization    -- input datasets['train_pool'], output datasets['train_preprocessing]
    b: feature selection                                            -- input datasets['train_preprocessing','selected_features','train_target'], output datasets['train_ready','selected_features'], to_csv('train_ready.csv','selected_features.csv')
step 5:
    a: split into train/development (7:3) datasets                  -- input datasets['train_ready','train_target'], output datasets['trainset','devset'], to_csv('trainset.csv','devset.csv')
step 6:
    a: svm model training and development desting                   -- input datasets['trainset','devset']




Note:
    - All .csv files are to be uploaded to Dropbox: team_share/data/

harry: work on 5a, evaluation, parameter tuning
sandy: work on 3z, 4a, generate dynamic.csv
taku: work 2a, 4b, other models
lawrence: refactor pipeline.py, generate train.csv/static.csv, other models
