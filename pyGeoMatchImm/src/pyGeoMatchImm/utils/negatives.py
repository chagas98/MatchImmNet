from ..utils.registry import register
from ..utils.base import PairsAnnotation, TrainConfigs, InputSequences, sSeq, sStruct, SamplePair, ChannelsInput
from ..utils.base import _AHO_CDR_RANGES, _PAIRS_ANNOTATION

import pandas as pd
import numpy as np
import json
import random
import Levenshtein
from typing import List, Dict, Any
from datetime import datetime
from itertools import product
import logging
log = logging.getLogger(__name__)

@register('DistNegativeSampler')
class DistNegativeSampler:
    def __init__(self, sample_pairs: List[SamplePair], proportion: int):

        self.sample_pairs = sample_pairs
        self.proportion = proportion
        self.cdr_ranges = _AHO_CDR_RANGES
        self.df = pd.DataFrame([])
        self.readable_cols = []

        self.pep_chain = PairsAnnotation.get_chain('epitope')
        self.MHCseq_chain = PairsAnnotation.get_chain('MHCseq')

        log.info(f'Preparating data for chains Peptide chain {self.pep_chain} and  MHC chain {self.MHCseq_chain}')
        log.info(f"Data: {len(self.sample_pairs)} positive samples.")

        self._prepare_data()
    
    def _prepare_data(self):

        ids_tcr,ids_pmhc, TCR_partial, peptide_partial, mhc_partial = [], [], [], [], []

        for pos_sample in self.sample_pairs:

            if pos_sample.label == 0:
                continue  # Skip negative samples
            
            # partial sequences
            ids_tcr.append(pos_sample.id)
            ids_pmhc.append(pos_sample.id)
            TCR_partial.append(''.join([pos_sample.get_cdr('CDR3A'), pos_sample.get_cdr('CDR3B')]))
            peptide_partial.append(pos_sample.getseq(chain=self.pep_chain[0]))
            mhc_partial.append(pos_sample.getseq(chain=self.MHCseq_chain[0]))

        self.df = pd.DataFrame({
            'id_tcr'      : ids_tcr,
            'id_pmhc'     : ids_pmhc,
            'pep_partial' : peptide_partial,
            'TCR_partial' : TCR_partial,
            'MHCseq'      : mhc_partial,
        })

        pep_order = self.df['pep_partial'].value_counts().index
        self.df['pep_partial'] = pd.Categorical(self.df['pep_partial'], categories=pep_order, ordered=True)
        self.df = self.df.sort_values('pep_partial').reset_index(drop=True)

    def _precompute_peptide_distances(self):
        unique_peptides = self.df['pep_partial'].unique()
        pep_pairing = pd.DataFrame(product(unique_peptides, repeat=2), columns=['col1', 'col2'])
        pep_pairing['ld'] = pep_pairing.apply(lambda row: Levenshtein.distance(row['col1'], row['col2']), axis=1)
        return pep_pairing

    def generate_negatives(self):
        pep_pairing = self._precompute_peptide_distances()
        available_partial_tcrs = list(np.repeat(self.df['TCR_partial'].values, self.proportion))
        list_new_partial_tcrs = []

        random.seed(42)  # For reproducibility
        np.random.seed(42)  # For reproducibility
        for i, row in self.df.iterrows():
            current_pep = row['pep_partial']
            #current_pep_complete = row['peptide']
            current_tcr = row['TCR_partial']

            # Find peptides with Levenshtein distance >= 3
            valid_peps = pep_pairing.query("col1 == @current_pep and ld >= 3")['col2'].unique()
            
            # Select TCRs that bind to these valid peptides ( >= 3 distance)
            possible_rows = self.df[self.df['pep_partial'].isin(valid_peps)]

            # Filter to only those TCRs that are still available
            possible_tcrs = possible_rows[possible_rows['TCR_partial'].isin(available_partial_tcrs)]['TCR_partial'].tolist()

            new_partial_tcr_list = []

            # If not enough possible TCRs, allow sampling from all available TCRs

            if len(possible_tcrs) < self.proportion:
                log.warning(f"Not enough distinct TCRs for peptide {current_pep} | {row['id_tcr']}. Sampling from all available TCRs.")

            pool = possible_tcrs if len(possible_tcrs) >= self.proportion else possible_rows['TCR_partial'].tolist()

            while len(new_partial_tcr_list) < self.proportion:
                sampled_tcr = np.random.choice(pool)
                if Levenshtein.distance(current_tcr, sampled_tcr) >= 3 and sampled_tcr not in new_partial_tcr_list:
                    new_partial_tcr_list.append(sampled_tcr)
                    if sampled_tcr in available_partial_tcrs:
                        available_partial_tcrs.remove(sampled_tcr)

            list_new_partial_tcrs.append(new_partial_tcr_list)

        return list_new_partial_tcrs

    def build_neg_dataset(self, list_new_partial_tcrs):
        flat_tcrs = [tcr for sublist in list_new_partial_tcrs for tcr in sublist]
        new_tcr_rows = []
        
        tcr_cols = ['id_tcr', 'TCR_partial']
        pmhc_cols = ['id_pmhc', 'pep_partial', 'MHCseq']

        for tcr in flat_tcrs:

            match_row = self.df[self.df['TCR_partial'] == tcr].iloc[0]
            new_tcr_rows.append(match_row[tcr_cols].values)

        new_dt_tcr = pd.DataFrame(new_tcr_rows, columns=tcr_cols)
        pmhc_data_x = self.df[pmhc_cols].loc[self.df.index.repeat(self.proportion)].reset_index(drop=True)
        self.new_dt_full_neg = pd.concat([new_dt_tcr.reset_index(drop=True), pmhc_data_x], axis=1)
        
        # Number the _neg for each repeat of TCR_ID
        self.new_dt_full_neg['id'] = self.new_dt_full_neg['id_tcr'] + '_' + self.new_dt_full_neg['id_pmhc']
        self.new_dt_full_neg['class'] = 0

        return self.new_dt_full_neg

    def parse_df_to_sp(self):
        """Create new negative samples (SamplePair) from the negative dataframe."""
        self.neg_sample_pairs = []
        for i, row in self.new_dt_full_neg.iterrows():
            for sp in self.sample_pairs:

                if sp.id == row['id_tcr']:
                    channel1_struct = sp.getstruct(list(_PAIRS_ANNOTATION.keys())[0])
                    channel1_seq = sp.getseq_channel(list(_PAIRS_ANNOTATION.keys())[0])

                if sp.id == row['id_pmhc']:
                    channel2_struct = sp.getstruct(list(_PAIRS_ANNOTATION.keys())[1])
                    channel2_seq = sp.getseq_channel(list(_PAIRS_ANNOTATION.keys())[1])
            
            if channel1_struct is None or channel2_struct is None:
                log.critical(f'Not possible get negative struct data for {row["id"]}. Skipping.')
            
            if channel1_seq is None or channel2_seq is None:
                log.critical(f'Not possible get negative seq data for {row["id"]}. Skipping.')

            new_sp = SamplePair(
                id=row['id'],
                channels=list(_PAIRS_ANNOTATION.keys()),
                struct=[channel1_struct, channel2_struct],
                seq=[channel1_seq, channel2_seq],
                label=0
            )
        
            self.neg_sample_pairs.append(new_sp)
        
        return self.neg_sample_pairs