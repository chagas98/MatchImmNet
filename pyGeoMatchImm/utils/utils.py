#! /usr/bin/env python3
import multiprocessing as mp
import pandas as pd
import numpy as np
from os import cpu_count
import logging
log = logging.getLogger(__name__)


def multiproc(run_function, data, num_workers=None):
    if num_workers is None:
        num_workers = min(12, max(1, int(0.7 * mp.cpu_count())))

    ctx = mp.get_context("fork")
    data = list(data)
    chunksize = max(1, len(data) // (num_workers * 8))

    results = []
    with ctx.Pool(processes=num_workers, maxtasksperchild=50) as pool:
        for out in pool.imap_unordered(run_function, data, chunksize=chunksize):
            results.append(out)  # ideally: save to disk here, not append
    return results



def validate_data(df: pd.DataFrame, dict_lists: dict, cols_to_check: list) -> bool:
    """
    Validate the input dataframe based on the provided criteria.

    df: Input dataframe to validate.
    dict_lists: Dictionary with column names as keys and lists of valid values as values.

    Returns True if validation passes, False otherwise.
    """
    
    df_lists = pd.DataFrame(dict_lists)
    df_lists.to_csv("temp_validation_lists.csv", index=False)

    log.info("Starting data validation...")
    for col in cols_to_check:
        if col not in df.columns:
            log.error(f"Missing column in input dataframe: {col}")
            return False
        if col not in df_lists.columns:
            log.error(f"Missing column in validation lists: {col}")
            return False

        for i, row in df_lists.iterrows():
            
            raw_id = row['id']
            
            if '_' in raw_id:
                id = raw_id.split('_')[0]
                id_neg = raw_id.split('_')[1]
            else:
                id = raw_id
                id_neg = None


            # Check for negative IDs
            if id_neg:

                if col == 'label':
                        if row[col] == 0:
                            continue
                        else:
                            log.error(f"Wrong Label for row {i}: id={raw_id}, col_value={row[col]}, col={col}")
                            return False
                
                # Negative
                if col in ['epitope', 'MHCseq', 'MHCa', 'MHCseq_ref']:
                    id_row_input = df[df['id'] == id_neg]

                    if id_row_input[col].values[0] == row[col]:
                        continue
                    else:
                        log.error(f"Validation failed for negative row {i}: id={raw_id}, col_value={row[col]}, col={col}")
                        log.error(f"Expected negative id: {id_row_input['id'].values[0]}, col_value: {id_row_input[col].values[0]}")
                        return False
            
                else:
                    id_row_input = df[df['id'] == id]
                    if id_row_input[col].values[0] == row[col]:
                        continue
                    else:
                        log.error(f"Validation failed for negative row {i}: id={raw_id}, col_value={row[col]}, col={col}")
                        log.error(f"Expected different value for id: {id_row_input['id'].values[0]}, col_value: {id_row_input[col].values[0]}")
                        return False
            
            # Check for positive IDs
            id_row_input = df[df['id'] == id]

            if id_row_input[col].values[0] == row[col]:
                continue
            else:
                log.error(f"Validation failed for row {i}: id={raw_id}, col_value={row[col]}, col={col}")
                log.error(f"Expected positive id: {id_row_input['id'].values[0]}, col_value: {id_row_input[col].values[0]}")
                return False


    log.info('Validation passed!')
    return True
