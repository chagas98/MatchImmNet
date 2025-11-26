#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any
from pydantic import BaseModel
import pandas as pd
import torch
from torch.utils.data import Dataset as Dataset_n

_PAIRS_ANNOTATION = {
    'TCR'  : {'TRA'    : 'D', 'TRB'   : 'E'},
    'pMHC' : {'epitope': 'C', 'MHCseq': 'A'}
}

_STRUCTURE_FILEPATH = {
    'TCR'  : 'filepath_a',
    'pMHC' : 'filepath_b'
}

_RSA_ENABLED_CHAINS = {
    'A': True,
    'C': False,
    'D': True,
    'E': True
}

_MHC_RES_SELECTION = [  #updated from new analysis of mhc renumbered structures
    19, 44, 55, 56, 58, 59, 61, 62, 63, 
    64, 65, 66, 68, 69, 70, 71, 72, 73, 
    75, 76, 79, 80, 83, 84, 108, 131, 145, 
    146, 147, 148, 149, 150, 151, 152, 153, 
    154, 155, 156, 157, 158, 159, 161, 162, 
    163, 166, 167, 169, 170]

_TCR_RES_SELECTION = {
    "E": {
      "CDR1a": list(range(27, 41)),
      "CDR2a": list(range(57, 71)),
      "CDR3a": list(range(109, 138)),
      "outCDRa": [84]
    },
    "D": {
      "CDR1b": list(range(30, 41)),
      "CDR2b": list(range(57, 71)),
      "CDR3b": list(range(109, 138)),
      "outCDRb": [85]
    }
  }

_AHO_CDR_RANGES = {
    "CDR1": list(range(25, 41)),
    "CDR2": list(range(58, 78)),
    "CDR3": list(range(109, 138)),
    "CDR2.5": list(range(83, 89))
}

RES_NAMES = [
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL'
]

RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYV'

TO_1LETTERS = {aaa:a for a,aaa in zip(RES_NAMES_1,RES_NAMES)}
TO_3LETTERS = {a:aaa for a,aaa in zip(RES_NAMES_1,RES_NAMES)}

ATOM_NAMES = [
    ("N", "CA", "C", "O", "CB"),                                                               # ala
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),                         # arg
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),                                           # asn
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),                                           # asp
    ("N", "CA", "C", "O", "CB", "SG"),                                                         # cys
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),                                     # gln
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),                                     # glu
    ("N", "CA", "C", "O"),                                                                     # gly
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),                             # his
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),                                          # ile
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),                                           # leu
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),                                       # lys
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),                                             # met
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),                       # phe
    ("N", "CA", "C", "O", "CB", "CG", "CD"),                                                   # pro
    ("N", "CA", "C", "O", "CB", "OG"),                                                         # ser
    ("N", "CA", "C", "O", "CB", "OG1", "CG2"),                                                 # thr
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"), # trp
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),                 # tyr
    ("N", "CA", "C", "O", "CB", "CG1", "CG2")                                                  # val
]
        
idx2ra = {(RES_NAMES_1[i],j):(RES_NAMES[i],a) for i in range(20) for j,a in enumerate(ATOM_NAMES[i])}

aa2idx = {(r,a):i for r,atoms in zip(RES_NAMES,ATOM_NAMES) 
          for i,a in enumerate(atoms)}
aa2idx.update({(r,'OXT'):3 for r in RES_NAMES})

@dataclass
class PairsAnnotation:
    annotation  = _PAIRS_ANNOTATION
    files       = _STRUCTURE_FILEPATH
    rsa_enabled = _RSA_ENABLED_CHAINS

    @classmethod
    def get_pairs(cls, channel_name: str) -> Dict[str, str]:
        return cls.annotation.get(channel_name, {})

    @classmethod
    def get_chains(cls, channel_name: str) -> Dict[str, bool]:
        return list(cls.get_pairs(channel_name).values())
    
    @classmethod
    def get_chain(cls, chain_name: str) -> str:

        for ch, names in cls.annotation.items():
            for name, chain in names.items():
                if name == chain_name:
                    return chain
        return ""
    @classmethod
    def get_chain_names(cls, channel_name: str) -> List[str]:
        return list(list(cls.get_pairs(channel_name).keys()))

    @classmethod
    def filetype(cls, channel_name: str) -> str:
        return cls.files.get(channel_name, "")

    @classmethod
    def is_chain_rsa_enabled(cls, chain: str) -> bool:
        return cls.rsa_enabled.get(chain, False)


class TrainConfigs(BaseModel):
    source          : Optional[str]
    channels        : Optional[List[str]] = ['TCR', 'pMHC']
    negative_prop   : Optional[int] = 1
    graph_method    : Optional[str]
    embed_method    : Optional[List[str]] = ["esm3"]
    edge_params     : Optional[List[str]]
    node_params     : Optional[List[str]]
    graph_params    : Optional[List[str]]
    other_params    : Optional[Dict[str, Any]]
    concat_embed    : Optional[str] = "all"
    norm            : Optional[bool] = False
    shuffle         : Optional[bool] = True
    drop_last       : Optional[bool] = True
    train_params    : Optional[Dict[str, Any]] = {}
    model_params    : Optional[Dict[str, Any]] = {}
    save_dir        : Optional[str] = None


class TestConfigs(BaseModel):
    batch_size : int
    num_epochs : int


@dataclass(order=True, eq=True)
class InputSequences:
    id             : str
    TRA            : str
    TRB            : str
    epitope        : str
    mhc_allele     : str
    MHCseq         : str
    filepath_a     : str
    filepath_b     : str
    label          : int  # or bool / Literal[0, 1]
    general        : Optional[str] = None                            # general proteins
    source         : Optional[str] = None                            # source of the data (e.g., pdb, tcr3d, etc.)
    MHCseq_ref     : Optional[str] = None                            # reference MHC sequence
    MHCseq_num     : tuple[int, ...] = field(default_factory=tuple)  # MHC sequence numbering
    TRA_ref        : Optional[str] = None                            # reference TRA sequence
    TRA_num        : tuple[int, ...] = field(default_factory=tuple)  # TRA sequence numbering
    TRB_ref        : Optional[str] = None                            # reference TRB sequence
    TRB_num        : tuple[int, ...] = field(default_factory=tuple)  # TRB sequence numbering
    CDR1A          : Optional[str] = None                            # CDR1 sequence of chain alpha
    CDR2A          : Optional[str] = None                            # CDR2 sequence of chain alpha
    CDR3A          : Optional[str] = None                            # CDR3 sequence of chain alpha           
    CDR1B          : Optional[str] = None                            # CDR1 sequence of chain beta    
    CDR2B          : Optional[str] = None                            # CDR2 sequence of chain beta   
    CDR3B          : Optional[str] = None                            # CDR3 sequence of chain beta

    @property
    def get_cdrs_beta(self) -> Dict[str, str]:
        cdrs = {}
        if self.CDR1B is not None:
            cdrs['CDR1B'] = self.CDR1B
        if self.CDR2B is not None:
            cdrs['CDR2B'] = self.CDR2B
        if self.CDR3B is not None:
            cdrs['CDR3B'] = self.CDR3B
        return cdrs

    @property
    def get_cdrs_alpha(self) -> Dict[str, str]:
        cdrs = {}
        if self.CDR1A is not None:
            cdrs['CDR1A'] = self.CDR1A
        if self.CDR2A is not None:
            cdrs['CDR2A'] = self.CDR2A
        if self.CDR3A is not None:
            cdrs['CDR3A'] = self.CDR3A
        return cdrs

    @property
    def sequences(self) -> Dict[str, str]:
        sequences = {
            "TRA": self.TRA,
            "TRB": self.TRB,
            "epitope": self.epitope,
            "MHCseq": self.MHCseq, 
            "CDR1A": self.CDR1A,
            "CDR2A": self.CDR2A,
            "CDR3A": self.CDR3A,
            "CDR1B": self.CDR1B,
            "CDR2B": self.CDR2B,
            "CDR3B": self.CDR3B
        }

        # exclude None values
        sequences = {k: v for k, v in sequences.items() if v is not None}

        return sequences

@dataclass
class sSeq:
    id           : str
    channel      : str
    chains       : Optional[List[str]] = field(default_factory=list)
    sequence     : Optional[Dict[str, str]] = field(default_factory=dict)
    sequence_ref : Optional[Dict[str,str]] = field(default_factory=dict)
    numbering    : Optional[Dict[str,str]] = field(default_factory=dict)
    cdrs         : Optional[Dict[str,str]] = field(default_factory=dict)
    
    def add_chain(self, chain_name: str, sequence: str, 
                  sequence_ref: Optional[str] = None, 
                  numbering: Optional[Dict[str,str]] = None,
                  cdrs: Optional[Dict[str,str]] = None) -> None:
        
        if self.chains is None:
            self.chains = []
        self.chains.append(chain_name)
        
        if self.sequence is None:
            self.sequence = {}
        self.sequence[chain_name] = sequence
        
        if sequence_ref:
            if self.sequence_ref is None:
                self.sequence_ref = {}
            self.sequence_ref[chain_name] = sequence_ref
        if numbering:
            self.numbering[chain_name] = numbering
        if cdrs:
            if self.cdrs is None:
                self.cdrs = {}
            self.cdrs.update(cdrs)


@dataclass
class sStruct:
    """3D structure info for a complete TCR–pMHC sample."""
    id        : str
    channel   : str
    chains    : List[str]
    filepath  : str
    str_info  : Dict[str, Any] = field(default_factory=dict) # chain: structure info to build graph
    graph     : Dict[str, Any] = field(default_factory=dict) # graph: any, nodes: any, edges: any


@dataclass
class SamplePair:
    id       : str
    channels : List[str]     = field(default_factory=list)
    struct   : List[sStruct] = field(default_factory=list)
    seq      : List[sSeq]    = field(default_factory=list)
    label    : Optional[int] = None

    def getstruct(self, name: str) -> sStruct:
        """Get sStruct data for a specific channel by name."""
        return next((s for s in self.struct if s.channel == name), None)
    
    def getseq_channel(self, name: str) -> sSeq:
        """Get sSeq data for a specific channel by name."""
        return next((s for s in self.seq if s.channel == name), None)
    
    def getseq(self, chain: str = None, channel: str = None) -> str | dict:
        """Get sequence for a specific chain | channel by name."""
        if chain:
            return next((s.sequence[chain] for s in self.seq if chain in s.chains), None)
        if channel:
            for s in self.seq:
                if s.channel == channel:
                    return s.sequence
        
        return None
    
    def get_cdr(self, name: str) -> str:
        """Get CDR sequence for a specific CDR name."""
        for s in self.seq:
            if name in s.cdrs:
                return s.cdrs[name]
        return None

    def get_allseqs(self) -> Dict[str, str]:
        """Get all sequences in the sample."""
        final_sequences = {}
        for s in self.seq:
            final_sequences.update(s.sequence)
        return final_sequences

    def getstrinfo(self, name: str) -> Any:
        """Get structure info for a specific channel by name."""
        return self.getstruct(name).str_info

    def getgraph(self, name: str) -> Any:
        """Get graph data for a specific channel by name."""
        return self.getstruct(name).graph

    def getgraphs(self) -> Dict[str, Any]:
        """Get all graph data in the sample."""
        return {s.channel: s.graph for s in self.struct}
    
    def get_cdrs(self) -> Dict[str, str]:
        """Get CDR sequences in the sample."""
        cdrs = {}
        for s in self.seq:
            for chain in s.chains:
                for key in s.sequence.keys():
                    if key.startswith("CDR"):
                        if s.sequence[key] is not None:
                            cdrs[key] = s.sequence[key]
        return cdrs
    
    def updatestructgraph(self, name: str, graph: Any) -> None:
        """Update graph data for a specific channel by name."""
        struct = self.getstruct(name)
        if struct:
            struct.graph = graph
        else:
            raise ValueError(f"Channel {name} not found in sample {self.id}.")
        

@dataclass(order=True)
class ChannelsInput:
    ids             : list[str]
    labels          : Optional[List[torch.tensor]] = None
    struct_channels : dict[str, List[Dataset_n]] = field(default_factory=dict)
    seq_channels    : dict[str, dict] = field(default_factory=dict)
    
