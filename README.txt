Ideas / Questions:
    - Does Kaggle submission accuracy increase as our devset accuracy increases?
    - What are the unquoted-but-bought plans like? Can they be defined as "neighboring" plans?
    - How different are the customers who buy unquoted plans than the "normal" customers?
    - Try more models
    - How to merge results from different models?
    - Label records with bought plans as 1 even if its record_type == 0
=========================================================================================================================================================================

/*******************************************************
**     NAMETAG = 'trainset' or 'devset0' or 'test'    **
**     INFOTAG = 'trainset' or 'train'                **
*******************************************************/

step 1: 
    a: read original data set train.csv                                 -- output datasets['train']

    b: read original data set test.csv                                  -- output datasets['test']
    
    z: split train.csv into trainset/devset0/devset1 (train:dev=7:3)    -- input datasets['train']
                                                                        -- output datasets['trainset','devset0','devset1'] 
                                                                        -- to_csv('trainset','devset0','devset1')

step 2:
    a: filter customers with changed characteristics                    -- input datasets[NAMETAG]
                                                                        -- output datasets[NAMETAG]

    b: discretize/recode some original features                         -- input datasets[NAMETAG,INFOTAG]
                                                                        -- output datasets[NAMETAG]
                                                                        -- to_csv(NAMETAG+'.csv')

step 3:
    a: create static features                                           -- input datasets[NAMETAG,INFOTAG]
                                                                        -- output datasets[NAMETAG+'_static'] 
                                                                        -- to_csv(NAMETAG+'_static.csv')

    b: create dynamic features                                          -- input datasets[NAMETAG] 
                                                                        -- output datasets[NAMETAG+'_dynamic'] 
                                                                        -- to_csv(NAMETAG+'_dynamic.csv')
                                                                        
    z: merge datasets, drop columns, dedup, split ID/features/target    -- input datasets[NAMETAG,NAMETAG+'_static',NAMETAG+'_dynamic'] 
                                                                        -- output datasets[NAMETAG+'_pool',NAMETAG+'_target',NAMETAG+'_customer_ID'] 
                                                                        -- to_csv(NAMETAG+'_pool.csv',NAMETAG+'_target.csv',NAMETAG+'_customer_ID.csv')

step 4:
    a: preprocess train/test pool, handle missing, normalization        -- input datasets[NAMETAG+'_pool','trainset' or 'dataset_w_mean_normalization_par_of_trainset'] 
                                                                        -- output datasets[NAMETAG+'_preprocessing']
                                                                        -- to_csv(NAMETAG+'_preprocessing.csv')
                                                                        
    b: feature selection                                                -- input datasets[NAMETAG+'_preprocessing','selected_features',NAMETAG+'_target'] 
                                                                        -- output datasets[NAMETAG+'_ready','selected_features'] 
                                                                        -- to_csv(NAMETAG+'_ready.csv','selected_features.csv')

step 5:


step 6:
    a: svm model training                                               -- input datasets['trainset_ready']
    b: svm model development/final testing                              -- input datasets[('devset0_ready','devset0_customer_ID', 'devset1')
                                                                                            or ('test_ready','test_customer_ID')]
    
step 7:
    a: post-training analysis (pull out missed cases) 

    b: benchmark

=========================================================================================================================================================================
There are 3 exec_seq:
    - trainset: ['1a','1z','2a','2b'train,'3a'train,'3b','3z','4a'train,'4b'train,'6a']
    - devset:   ['1a','1z','2a','2b'test,'3a'test,'3b','3z','4a'test,'4b'test,'6a']
    - test:     ['1b','2a','2b'test,'3a'test,'3b','3z','4a'test,'4b'test,'8a']

!!! Note that we should process devset/test exactly the same way as we process trainset, which means:
    - Some feature generation requires hash table lookup; these hash tables should be based on information from trainset.
    - When we recode/fillna/normalize features, the cutoffs/mean/min/max etc. should be based on information from trainset.
    - Feature selection is only done on trainset. The same set of features should be kept for devset/test.  
    - Some functions should be executed in 2 modes: train or test. They include 2b,3a,4a,4b for now. We may add more to this list.

=========================================================================================================================================================================
Data Location:
    - All .csv files are to be uploaded to Dropbox: team_share/data/


