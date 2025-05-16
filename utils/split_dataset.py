import pandas as pd
import numpy as np
import re

def shuffle_groups(df, group_column, SEED):
    # Get unique groups and shuffle them
    shuffled_groups = np.random.RandomState(seed=SEED).permutation(df[group_column].unique())
    
    # Reorder the DataFrame based on the shuffled groups
    reordered_df = pd.concat([df[df[group_column] == group] for group in shuffled_groups])
    
    return reordered_df.reset_index(drop=True)

def get_idx(df, start_idx, set_len):
    
    end_row = df.iloc[start_idx + set_len, :]
    end_group = end_row['group']
    end_idx = end_row.name
    
    end_group_idx = df['group'][df['group'] == end_group].index.to_list()

    if (end_idx - min(end_group_idx)) >= 0.5*len(end_group_idx):
        return max(end_group_idx)
    else:
        return min(end_group_idx) - 1

def split(db_file, SEED, SPLIT_RATIO, MIXED = True):
    
    db = pd.read_csv(db_file)

    SPLIT_RATIO = re.split(',', SPLIT_RATIO)
    SPLIT_RATIO = list(map(float, SPLIT_RATIO)) 

    if MIXED == False: # len(SPLIT_RATIO) = 2: train/val sets are peptides and split; test set are polymers
    
        peptides_df = db[db["ID"].str.contains("pep", na=False)]
        peptides_df = peptides_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        train_idx = int(len(peptides_df) * SPLIT_RATIO[0])
        train_set = peptides_df.iloc[:train_idx]
        val_set   = peptides_df.iloc[train_idx:]

        polymer_df = db[db["ID"].str.contains("poly", na=False)]
        test_set = polymer_df

    if MIXED == True: # len(SPLIT_RATIO) = 3: mixes peptides + polymers to create train/val/test sets
        
        poly_idx = db['ID'].str.contains('poly')
        db['group'] = db[poly_idx]['ID'].apply(lambda x: int(re.split(r'_', x)[0][6:]))
    
        start_idx = max(db['group'].value_counts().index.to_list())
    
        vals = list(db['group'].value_counts().to_list())
        
        if np.unique(vals).size == 1:
            k = vals[0]
    
        pep_idx = db['ID'].str.startswith('pepID')
    
        # Shuffle and assign groups directly
        db.loc[pep_idx, 'group'] = (
            db[pep_idx]
            .sample(frac=1, random_state=SEED)  # Shuffle only the matching rows
            .assign(group=lambda x: (np.arange(len(x)) // k) + start_idx + 1)['group']  # Group by size 2
        )
    
        db['group'] = db['group'].astype(int) 
    
        df_shuffled = shuffle_groups(db, 'group', SEED)
    
        # train set
        train_start_idx = 0
        train_set_len = int(np.around(len(df_shuffled)*SPLIT_RATIO[0]))
        train_end_idx = get_idx(df_shuffled, train_start_idx, train_set_len) + 1
        train_set = df_shuffled.iloc[train_start_idx:train_end_idx,:]
    
        # val set
        val_start_idx = train_end_idx
        val_set_len = int(np.around(len(df_shuffled)*SPLIT_RATIO[1]))
        val_end_idx = get_idx(df_shuffled, val_start_idx, val_set_len) + 1
        val_set = df_shuffled.iloc[val_start_idx:val_end_idx,:]
    
        # test set
        test_start_idx = val_end_idx
        test_set_len = int(np.around(len(df_shuffled)*SPLIT_RATIO[2]))
        test_end_idx = get_idx(df_shuffled, test_start_idx, test_set_len) + 1
        test_set = df_shuffled.iloc[test_start_idx:test_end_idx,:]

    return  {'mixed': MIXED, 
             'train': train_set.set_index('ID')['sequence'].to_dict(), 
             'val': val_set.set_index('ID')['sequence'].to_dict(), 
             'test': test_set.set_index('ID')['sequence'].to_dict()}
    
    