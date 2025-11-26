#!/usr/bin/env python3
import sys
import os
import pandas as pd
import json
import argparse
import datetime
from pyMatchImm.datasets import TCR3dataset, VDJdbdataset, McPASTCRdataset, IEDBdataset, NeoTCRdataset, PosTRAITdataset, TESTdataset, ConcatDataset

#set the args
parser = argparse.ArgumentParser(description='Run get_datasets script to load and concatenate TCR datasets.')
parser.add_argument("-i", "--datasets_list", nargs='+', help="choose datasets to include", type=str)
parser.add_argument("-l", "--datasets_dir", help="location of public datasets", type=str)
parser.add_argument("-r", "--ref_mhc", help="location of reference MHC file", type=str)
parser.add_argument("-m", "--map_datasets", help="map dataset file names", type=json.loads, 
                    default='{"vdjdb": "vdjdb_human_monkey_mouse_TRA_TRB_paired_MHCI_score0_030625.tsv", \
                            "mcpas": "McPAS_TCR_updatedSep2022_10062025.csv", \
                            "iedb": "iedb_raw_linear_onlypositive_MHCI_TCRpaired.csv", \
                            "neo": "NeoTCR data-20221220.xlsx", \
                            "trait": "20250312-TRAIT_search_download.xlsx", \
                            "tcr3d_metadata": "tcr3d_classI_complexes_2025jun22.csv", \
                            "tcr3d_struct": "TCR3d_complexes" \
                            }', required=False)
parser.add_argument("--all", default=False, action='store_true')
parser.add_argument("-d", "--description", help="description of the dataset", type=str)
parser.add_argument("-s", "--score_threshold", help="score threshold for SOME datasets", type=float, default=0.0)
args = parser.parse_args()

# PARAMETERS
DATASETS_LIST = args.datasets_list
LOCATION      = args.datasets_dir
REF_MHC       = args.ref_mhc
MAP_DATASETS  = args.map_datasets
RUN_ALL       = args.all
SCORE         = args.score_threshold
DESCRIPTION   = args.description

print("-" * 50 +
      f"\nRunning get_datasets with parameters:\n"
      f"Datasets List: {DATASETS_LIST}\n"
      f"Location: {LOCATION}\n"
      f"Reference MHC: {REF_MHC}\n"
      f"Map Datasets: {MAP_DATASETS}\n"
      f"Run All: {RUN_ALL}\n"
      f"Score Threshold: {SCORE}\n"
      f"Description: {DESCRIPTION}\n" +
      "-" * 50)


def resolve_mapping(mapping):
    for key in list(mapping.keys()):
        while mapping.get(mapping[key]):
            mapping[key] = mapping[mapping[key]]
    return mapping

def find_equal_seqs(df, column):
    sequences = df[column].unique()
    equal_seqs = {}

    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            if i == j:
                continue

            # If one is a substring of the other
            if seq1 in seq2 and len(seq2) > len(seq1):
                # Always map longer to shorter
                shorter, longer = (seq1, seq2)

                if shorter in equal_seqs.keys():
                    shorter = equal_seqs[shorter]

                if longer not in equal_seqs.keys():
                    equal_seqs[longer] = shorter
                else:
                    # Keep the shortest seen so far
                    shorter_prev = equal_seqs[longer]
                    equal_seqs[longer] = min(shorter_prev, shorter, key=len)
                    equal_seqs[shorter] = min(shorter_prev, shorter, key=len)

    
    if 'ref' not in column:
        # Replace based on mapping
        equal_seqs = resolve_mapping(equal_seqs)
        df[f'{column}_ref'] = df[column].replace(equal_seqs)
    else:
        df[column] = df[column].replace(equal_seqs)
    return df

def get_datasets():
    """ Load and concatenate datasets for TCR analysis.
    
    This function loads multiple TCR datasets, processes them, and returns a concatenated DataFrame.
    It also handles the mapping of columns for the test dataset.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of all datasets.
    """

    # Map Test Dataset Columns
    map_cols = {
        'CDR3a': 'CDR3A',
        'CDR3b': 'CDR3B',
        'peptide': 'epitope',
        'Va': 'VA',
        'Vb': 'VB',
        'Ja': 'JA',
        'Jb': 'JB',
        'MHCa': 'MHCa',
        'class': 'class'
    }
    
    datasets = {}
    
    if 'vdjdb' in DATASETS_LIST:
        vdjdb_file = f'{LOCATION}/{MAP_DATASETS["vdjdb"]}'
        dataVDJ = VDJdbdataset(path=vdjdb_file, score_threshold=SCORE)
        datasets['vdjdb'] = dataVDJ
    
    if 'mcpas' in DATASETS_LIST:
        mcpas_file = f'{LOCATION}/{MAP_DATASETS["mcpas"]}'
        dataMcPAS = McPASTCRdataset(path=mcpas_file)
        datasets['mcpas'] = dataMcPAS
    
    if 'iedb' in DATASETS_LIST:
        iedb_file = f'{LOCATION}/{MAP_DATASETS["iedb"]}'
        dataIEDB = IEDBdataset(path=iedb_file)
        datasets['iedb'] = dataIEDB
    
    if 'neo' in DATASETS_LIST:
        neo_file = f'{LOCATION}/{MAP_DATASETS["neo"]}'
        dataNeo = NeoTCRdataset(path=neo_file)
        datasets['neo'] = dataNeo
    
    if 'trait' in DATASETS_LIST:
        trait_file = f'{LOCATION}/{MAP_DATASETS["trait"]}'
        dataTrait = PosTRAITdataset(path=trait_file)
        datasets['trait'] = dataTrait
    
    if 'tcr3d' in DATASETS_LIST:
        tcr3d_metadata_path = f'{LOCATION}/{MAP_DATASETS["tcr3d_metadata"]}'
        tcr3d_meta_df = pd.read_csv(tcr3d_metadata_path)
        tcr3d_struct_path = f'{LOCATION}/{MAP_DATASETS["tcr3d_struct"]}'

        tmp = 'tmp/'
        os.makedirs(tmp, exist_ok=True)

        dataTCR3d = TCR3dataset(pdb_dir       = tcr3d_struct_path,
                                suffix        = '.trunc.fit.pdb',
                                from_fasta    = True,
                                ref_mhc       = REF_MHC,
                                selection_ids = tcr3d_meta_df['PDB ID'].tolist())

        datasets['tcr3d'] = dataTCR3d

    if 'test' in DATASETS_LIST:
        test_file = f'{LOCATION}/test_{DESCRIPTION}.csv'
        dataTest = TESTdataset(path=test_file, map_cols=map_cols, outfile=f'test_{DESCRIPTION}.csv')
        datasets['test'] = dataTest

    concat_data = ConcatDataset(datasets = list(datasets.values()),
                                labels   = list(datasets.keys()))
    df = concat_data.to_df()
    print(f"Concatenated dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    print(50*"-")
    cdate = datetime.datetime.now()
    dataset_name = '_'.join(DATASETS_LIST)
    outfile_name = f'{dataset_name}_{cdate.year}{cdate.month:02d}{cdate.day:02d}.csv'
    df.to_csv(outfile_name, index=False)

    return df, outfile_name

def parse_to_pyGeoMatchImm(df, outfile_name):

    map_rename = {
        'TCR_ID'         : 'id',
        'TRA'            : 'TRA',
        'TRB'            : 'TRB',
        "CDR1A"          : 'CDR1A',
        "CDR2A"          : 'CDR2A',
        "CDR3A"          : 'CDR3A',
        "CDR1B"          : 'CDR1B',
        "CDR2B"          : 'CDR2B',
        "CDR3B"          : 'CDR3B',
        'TRA_ref'        : 'TRA_ref',
        'TRB_ref'        : 'TRB_ref',
        'TRA_num'        : 'TRA_num',
        'TRB_num'        : 'TRB_num',
        'peptide'        : 'epitope',
        'MHCseq'         : 'MHCseq',
        'MHCseq_ref'     : 'MHCseq_ref',
        'mhc_allele'     : 'mhc_allele',
        'filepath_a'     : 'filepath_a',
        'filepath_b'     : 'filepath_b',
        'label'          : 'label',
        'source'         : 'source'
    }

    df.dropna(subset=['TCR_ID', 'TRA', 'TRB', 'MHCseq', 'peptide'], inplace=True)
    df['filepath_a']   = df['PDB_ID'].apply(lambda x: os.path.join(LOCATION,'TCR3d_complexes/TCR', x + '.trunc.fit_split.pdb') if not x.startswith('/') else x)
    df['filepath_b']   = df['PDB_ID'].apply(lambda x: os.path.join(LOCATION,'TCR3d_complexes/pMHC_renum', x + '.trunc.fit_split_renum.pdb') if not x.startswith('/') else x)
    df['source']     = df['Release_date']
    df['mhc_allele'] = df.apply(lambda row: row['allele_blast'] if not row['allele'] else row['allele'], axis=1)
    df['label']      = 1
    
    print("Finding equal sequences in TRA, TRB, and MHCseq_ref columns...")
    print("TRA sequences")
    df = find_equal_seqs(df, column='TRA')
    print("TRB sequences")
    df = find_equal_seqs(df, column='TRB')
    print("MHCseq_ref sequences")
    df = find_equal_seqs(df, column='MHCseq_ref')

    df.rename(columns=map_rename, inplace=True)
    df = df[list(map_rename.values())]

    df.to_csv(outfile_name.split('.')[0] + '_renamed.csv')


if __name__ == "__main__":

    if RUN_ALL:
        df, outfile_name = get_datasets()
        parse_to_pyGeoMatchImm(df, outfile_name)
    else:
        #find local .csv file
        path = 'tcr3d_20251004.csv'
        if path:
            df = pd.read_csv(path)
            parse_to_pyGeoMatchImm(df, path)

else:
    print("This script is intended to be run as a standalone program.")
    print("Please execute it directly to run the preparation pipeline.")

# python3 prepare_data.py -i tcr3d -l /home/samuel.assis/MatchImm/Public_Datasets  --ref_mhc /home/samuel.assis/MatchImm/Public_Datasets/mhc/hla_prot_includeMusMusculus.fasta --all