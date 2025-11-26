#!/usr/bin/env python3

# local toolkit
from ..utils.registry import register, get
from ..utils.base import (SamplePair, PairsAnnotation,
                          _TCR_RES_SELECTION, _MHC_RES_SELECTION,
                          TrainConfigs 
                          )

# third-parties
import MDAnalysis as mda
import numpy as np
from itertools import chain
import logging
log = logging.getLogger(__name__)

def _unique_resids(u: mda.Universe, selection: str) -> list[int]:
    ag = u.select_atoms(selection)
    if ag.n_atoms == 0:
        return []
    return np.unique(ag.resids).astype(int).tolist()

# TODO IMPROVE INTERFACE RESIDUES SELECTION
def get_interface_tcrpmhc(pdb_file: str, channel_name) -> dict:

    u = mda.Universe(pdb_file)

    interface_positions = {}

    if channel_name == 'TCR':

        interface_positions['D'] = list(chain.from_iterable(list(_TCR_RES_SELECTION['D'].values())))
        interface_positions['E'] = list(chain.from_iterable(list(_TCR_RES_SELECTION['E'].values())))

    elif channel_name == 'pMHC':
        resids_epitope = _unique_resids(u,f"protein and (chainID C)") # all residues of epitope
        interface_positions['C'] = resids_epitope

        #resids_mhc = _unique_resids(u,f"protein and chainID A and byres around 5 (chainID C)") # around chain C (epitope)
        interface_positions['A'] = _MHC_RES_SELECTION

    else:
        raise ValueError(f"Unknown channel name '{channel_name}'")
    
    return interface_positions 

class StructureParser:
    def __init__(self, configs: TrainConfigs):
        self.cfg = configs

        parser_methods = {
            "graphein": self._graphein,
            "builtin" : self._builtin,
        }

        self._pmethod = parser_methods[self.cfg.graph_method]

    def __call__(self, sample: SamplePair) -> SamplePair:

        for ch_struct in sample.struct:
            
            log.debug(f"Parsing structure for sample {sample.id}, channel {ch_struct.channel}")
            
            channel_name = ch_struct.channel
            str_info = self._pmethod(channel_name, sample)
            ch_struct.str_info = str_info

        log.debug(f"Finished parsing structures for sample {sample}")

        return sample

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _graphein(channel_name: str, sample: SamplePair) -> dict:
        """
        Prepare data for Graphein processing.
        """

        ### ADD MULTIPLE FILE PATH FOR TCR - PMHC PREDICTED SEPARATELY
        filepath = sample.getstruct(channel_name).filepath

        return get_interface_tcrpmhc(filepath, channel_name)
        

    @staticmethod
    def _builtin(channel_name: str, chain: str, sample: SamplePair) -> dict:
        """
        Prepare data for built-in processing.
        """
        raise NotImplementedError("Builtin processor not implemented yet.")


#class mmCIFParser