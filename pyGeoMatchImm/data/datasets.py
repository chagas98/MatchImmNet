#!/usr/bin/env python3

# local toolkit
import networkx as nx
from networkx import nodes
from ..utils.registry import register, get
from ..utils.negatives import DistNegativeSampler
from ..utils.base import PairsAnnotation, TrainConfigs, InputSequences, sSeq, sStruct, SamplePair, ChannelsInput
from ..utils.utils import multiproc
from ..embeddings.esm import ESM3Embedder
from ..embeddings.atchley import AtchleyEmbedder
from .parser import StructureParser
from .generators.graphein import GrapheinGeneratorRes, GrapheinToChannels
from ..utils.visualize import print_pymol_resids, launch_pymol, save_graph

# torch toolkit
import torch
import torch.nn as nn
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torchvision import transforms

# third-parties
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np
import os
from os import cpu_count, name
from random import shuffle
import logging
log = logging.getLogger(__name__)

class BaseDataset:
    """
    Base class for all datasets.
    """
    def __init__(self, data_path: str, config: TrainConfigs):
        self.config    = config
        self.df_input  = pd.read_csv(data_path).head(30)

        log.info('Creating Dataset...\n')
        log.info(f'Number of samples: {len(self.df_input)}')
        # Filter the DataFrame to only include columns defined in InputSequences
        self.df = self.df_input[self.df_input.columns.intersection(InputSequences.__annotations__.keys())]
        # Check for missing structure files
        self._check_struct_file()

    def _check_struct_file(self):
        """
        Check if the structure files exist.
        """
        checking_a = [f for f in self.df['filepath_a'] if not os.path.exists(f)]
        checking_b = [f for f in self.df['filepath_b'] if not os.path.exists(f)]
        
        if checking_a or checking_b:
            raise FileNotFoundError(f"Structure files not found: {checking_a + checking_b}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> InputSequences:
        return InputSequences(**self.df.iloc[idx].to_dict())

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @property
    def to_df(self) -> pd.DataFrame:
        return self.df.copy()

@register("datasets.pdb")
class PDBDataset(BaseDataset):
    """Dataset for PDB structures."""
    def __init__(self, data_path: str, config: TrainConfigs):
        super().__init__(data_path, config)
        log.info("PDBDataset initialized")
        self.preprocess()

    def preprocess(self):
        return self.df

@register("datasets.model")
class ModelDataset(BaseDataset):
    """Dataset for model structures."""
    def __init__(self, data_path: str, config: TrainConfigs):
        super().__init__(data_path, config)
        log.info("ModelDataset initialized")

    def preprocess(self):
        # Implement any necessary preprocessing steps here
        # Processar predições AF3/TCRmodel2
        pass

@dataclass(order=True)
class TCRpMHCDataset:
    def __init__(self, data_path: str, config: dict):
        
        print("Initializing TCRpMHCDataset...")
        self.cfg = TrainConfigs(**config)

        # Data processing
        self._dataprocessing = get(f"datasets.{self.cfg.source}")  # PDBDataset or ModelDataset
        self._dataset_seq: BaseDataset = self._dataprocessing(data_path, self.cfg)
        
        # Structure parser
        self._structparser = StructureParser(self.cfg)
        
        # Graph generation
        self._graphgenerator = get(f"graph.{self.cfg.graph_method}")(**self.cfg.__dict__)# GrapheinGenerator
        

        log.info("\nCreating dataset...\n")
        log.info(f"Number of samples: {len(self._dataset_seq)}")
        log.info(f"Annotations:                   \
                    {PairsAnnotation.annotation}  \
                    {PairsAnnotation.files}       \
                    {PairsAnnotation.rsa_enabled}")

        log.info("\nBuilding Pairing samples based on channels...\n")
        self._paired = self._pairing()

        log.info("\nParsing structures...\n")
        # Structure parser
        self._parsed = self._struct_parser()

        log.info(f"\nGenerating graphs using {self.cfg.graph_method}...\n")
        # Featurization
        self._dataset = self._graph_generator()

        log.info(f"\nGenerate negatives\n")
        self._dataset_neg = self._genNegatives()

        log.info(f"\nCombining positives and negatives...\n")
        self._dataset.extend(self._dataset_neg)
        shuffle(self._dataset)

        self._seq_embedder() # ESM embedder

        log.info(f"IDs: {[ds.id for ds in self._dataset]} ...")
        log.info(f"Total samples after adding negatives: {len(self._dataset)}"
                 f" (Positives: {sum(sp.label == 1 for sp in self._dataset)}, "
                 f"Negatives: {sum(sp.label == 0 for sp in self._dataset)})")
        
        #TODO: add option to shuffle dataset
        
        # fetchers
        log.info("\nFetching data...\n")
        self._fetchers()


    def _get_channels(self, sample_seq: InputSequences) -> SamplePair:
        """
        Create paired SamplePair based on channels (PairsAnnotation).
        
        Args:
            sample_seq (InputSequences): Input sequence data.
        Returns:
            SamplePair: Paired sample with sequences and structures.
        """
        
        ch_names = self.cfg.channels
        log.debug(f"Processing sample {sample_seq.id} for channels {ch_names}")
        seq_list = []
        struct_list = []
        
        for ch_name in ch_names:
            
            # Get required domains, chains, and filetypes from a pre-defined annotation
            domains_required = PairsAnnotation.get_chain_names(ch_name)
            chains_required = PairsAnnotation.get_chains(ch_name)
            filepath_required = PairsAnnotation.filetype(ch_name)
            
            log.debug(f"Channel: {ch_name}, \
                        Chains: {chains_required}, \
                        Filepath: {filepath_required}")
            
            # Validate required fields
            self._validate_seq(sample_seq, require=tuple(domains_required))

            # Get the filepath to structure
            filepath = getattr(sample_seq, filepath_required)
            
            # Create sSeq and sStruct objects
            sseq = sSeq(id=sample_seq.id, channel=ch_name)
            sstruct = sStruct(id=sample_seq.id, channel=ch_name, chains=chains_required, filepath=filepath)

            for domain, chain in zip(domains_required, chains_required):

                # Get CDRs for TCR alpha or TCR beta chains
                if domain.upper() in ['TRA', 'ALPHA', 'TCRA']:
                    cdrs = sample_seq.get_cdrs_alpha
                elif domain.upper() in ['TRB', 'BETA', 'TCRB']:
                    cdrs = sample_seq.get_cdrs_beta

                # Log CDRs found
                if cdrs:
                    log.debug(f"CDRs found for {sample_seq.id} in channel {ch_name}: {list(cdrs.keys())}")                
                else:
                    cdrs = None # No CDRs for non-TCR domains

                sseq.add_chain(chain_name   = chain, 
                               sequence     = getattr(sample_seq, domain),
                               sequence_ref = getattr(sample_seq, f"{domain}_ref", None),
                               numbering    = getattr(sample_seq, f"{domain}_num", None),
                               cdrs         = cdrs if cdrs else None)

            seq_list.append(sseq)
            struct_list.append(sstruct)

        return SamplePair(id=sample_seq.id, channels=ch_names, seq=seq_list, struct=struct_list, label=sample_seq.label)


    def _pairing(self):
        """
        Create pairs of samples based on channels.
        """
        return [self._get_channels(sample) for sample in self._dataset_seq]

    def _struct_parser(self):
        """ Parse structures in the paired samples. Apply multiprocessing."""
        #return [self._parser(sample) for sample in self.paired] for debugging
        return multiproc(self._structparser, self._paired)
    
    def _seq_embedder(self):
        
        for method in self.cfg.embed_method:
            match method:
                case 'esm3':
                    log.info("Embedding sequences using ESM-3...")
                    self._embedder = ESM3Embedder(**self.cfg.__dict__)
                    self._embedder.init_model() # initialize model once

                case 'atchley':
                    log.info("Embedding sequences using Atchley factors...")
                    self._embedder = AtchleyEmbedder()
                
            for sample in self._dataset:
                struct_files = {struct.channel: struct.filepath for struct in sample.struct}
                graphs = sample.getgraphs()

                graphs_updated  = self._embedder(struct_files, graphs)
                for ch_name, g in graphs_updated.items():
                    sample.updatestructgraph(ch_name, g) # update graph in sample
                    log.debug(f"{method}: Updated graph for channel {ch_name} in sample {sample.id}:")

                
    def _genNegatives(self):
        sampler = DistNegativeSampler(
            sample_pairs=self._dataset,
            proportion=self.cfg.negative_prop
        )

        list_new_partial_tcrs = sampler.generate_negatives()
        sampler.build_neg_dataset(list_new_partial_tcrs)
        return sampler.parse_df_to_sp()

    def _graph_generator(self):
        """ Generate graphs for the parsed samples. Apply multiprocessing."""
        #return [self._generator(sample) for sample in self.parsed]
        return multiproc(self._graphgenerator, self._parsed)

    def _fetchers(self):
        """
        Directly fetch data for the dataset.
        """
        # IDs
        self.ids = [sample.id for sample in self._dataset]
        
        # Graphs
        self.graphs = {sample.id: sample.getgraphs() for sample in self._dataset}     
        
        # Sequences
        self.sequences = {sample.id: sample.get_allseqs() for sample in self._dataset}

        # Struct info
        self.structures = {sample.id: sample.struct for sample in self._dataset}

    def show_graph(self, sample_id: str, pymol_gui: bool = False):
        graphs = self.graphs.get(sample_id, None)
        if graphs:
            log.info(f"Graph details for sample {sample_id}:")
            log.info(f"Channels: {list(graphs.keys())}")

            pymol_input = {}
            for channel_name, graph in graphs.items():
                log.info(f"\n\n{50 * '-'}Channel {channel_name}{50 * '-'}\n\n")
            
                all_graph = graph
                for node in all_graph.nodes(data=True):
                    log.info(f"  Node {node[0]}: {node[1]}")

                for edge in all_graph.edges(data=True):
                    log.info(f"  Edge {edge[0]}-{edge[1]}: {edge[2]}")

                # Prepare Pymol Input
                filepath = [struct.filepath for struct in self.structures[sample_id] if struct.channel == channel_name][0]

                pymol_input[channel_name] = {}
                pymol_input[channel_name]['path'] = filepath
                pymol_input[channel_name]['resids'] = all_graph.nodes(data=True)
                pymol_input[channel_name]['interactions'] = all_graph.edges(data=True)

                # Save plot
                save_graph(all_graph, channel_name)

        else:
            log.warning(f"No graph found for sample {sample_id}")

        if pymol_gui:
            launch_pymol(pymol_input)
        else:
            print_pymol_resids(pymol_input)

    @staticmethod # explicitly declare as pure function, without self interaction
    def _validate_seq(obj: sSeq, require: tuple[str, ...]) -> None:
        missing = [name for name in require if getattr(obj, name, None) in (None, "")]
        if missing:
            raise ValueError(f"Missing required fields in sSeq: {missing}")
        
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> SamplePair:
        return self._dataset[idx]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class ChannelsGraph:
    def __init__(self, dataset: TCRpMHCDataset, config: dict, embed_method: str):
        log.info('Building Protein Features inputs...')

        self.cfg = TrainConfigs(**config)
        self.embed_method = embed_method
        self._convertor = get(f"channels.{self.cfg.graph_method}")(config=self.cfg, embed_method=self.embed_method) # GrapheinToChannels
        self._dataset = dataset

        results_topyg = multiproc(self._convertor, self._dataset)

        ids, labels, str_dataset, seq_dataset = zip(*results_topyg)
        self._str_dataset = str_dataset   
        self._seq_dataset = seq_dataset
        self.channel_names = list(self.cfg.channels)

        # ---- IDs & labels (keep two forms) ----
        self.ids = list(ids)
        self.y_list = list(labels)  # for sklearn
        self.y_tensor = torch.tensor(self.y_list, dtype=torch.float32).view(-1, 1)  # for torch

        # Pick first two structural channels deterministically
        assert len(self.channel_names) >= 2, "Need at least two channels"

        ch1_name, ch2_name = self.channel_names[:2]

        self.ch1 = (self.to_minimal(g) if g is not None else None
                    for g in self.iter_channel(self._str_dataset, ch1_name))
        self.ch2 = (self.to_minimal(g) if g is not None else None
                    for g in self.iter_channel(self._str_dataset, ch2_name))
        self.ids = list(ids)

    @staticmethod
    def iter_channel(dataset, key):
        return (sample.get(key, None) for sample in dataset)

    #-----------Graph Featurize----------
    def to_minimal(self, g: Data) -> Data:
        # keep only x and edge_index)
        if hasattr(g, self.embed_method):
            x = g[self.embed_method]
        else:
            raise ValueError(f"Graph does not have attribute '{self.embed_method}'")

        return Data(x=torch.as_tensor(x), edge_index=g.edge_index, 
                    chain_id=g.chain_id, resid=g.residue_number, 
                    resname=g.residue_name, name=g.name) #TODO structural information can be add here

    # ---------- Getters ----------
    def get_str_channel(self, name: str):
        return self.iter_channel(self._str_dataset, name) # create a NEW iterator, don’t reuse a stored generator

    def get_seq_channel(self, name: str):
        return self.iter_channel(self._seq_dataset, name) # create a NEW iterator, don’t reuse a stored generator

    def get_seq_chain(self, name: str, chain: str):
        return [(s.get(chain) if s is not None else None) for s in self.get_seq_channel(name)]

    def get_labels(self): 
        return self.y_list



class ChannelsPairDataset(Dataset_n):
    """
    Minimal two-channel dataset for PyG two-tower models.
    Expects each channel graph to have .x (node features) and .edge_index.
    Returns HeteroData with node types 'ch1' and 'ch2' + graph-level y.
    """
    def __init__(
        self,
        ids: List[str],
        ch1_graphs: Sequence[Data],
        ch2_graphs: Sequence[Data],
        labels: Union[Sequence[int], Sequence[float], torch.Tensor],
        ch1_name: str = "ch1",
        ch2_name: str = "ch2",
        y_dtype: torch.dtype = torch.float32,
        mask_between: bool = True,
        mask_chain_map: dict = {'A': False, 'C': True, 'D': True, 'E': True}
    ):
        self.ch1 = list(ch1_graphs)
        self.ch2 = list(ch2_graphs)
        self.n = len(self.ch1)
        self.ch1_name, self.ch2_name = ch1_name, ch2_name
        self.ids = ids

        self.mask_chain_map = mask_chain_map
        self.mask_between = mask_between

        #assert len(self.ch1) == len(self.ch2) == len(ids), "Channel lengths must match"
        
        if isinstance(labels, torch.Tensor):
            assert labels.shape[0] == self.n
            self.y = labels.reshape(-1, 1).to(y_dtype)
        else:
            self.y = torch.tensor(labels, dtype=y_dtype).view(-1, 1)

        # (Optional) sanity: ensure x exists and dims are consistent per type
        d1 = self.ch1[0].x.size(-1); d2 = self.ch2[0].x.size(-1)
        for g in self.ch1:
            assert hasattr(g, "x") and hasattr(g, "edge_index")
            assert g.x.size(-1) == d1, "All ch1.x must share same width"
        for g in self.ch2:
            assert hasattr(g, "x") and hasattr(g, "edge_index")
            assert g.x.size(-1) == d2, "All ch2.x must share same width"
    
    @staticmethod
    def _as_long_idx(idx, N: int) -> torch.Tensor:
        if isinstance(idx, slice):
            return torch.arange(*idx.indices(N), dtype=torch.long)
        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                idx = np.flatnonzero(idx)
            return torch.as_tensor(idx, dtype=torch.long).view(-1)
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                idx = idx.nonzero(as_tuple=False).view(-1)
            return idx.view(-1).long()
        if isinstance(idx, (list, tuple, range)):
            return torch.tensor(list(idx), dtype=torch.long)
        return torch.tensor([int(idx)], dtype=torch.long)  # scalar

    def y_at(self, idx) -> torch.Tensor:
        """Return labels for the given indices as a [k, 1] tensor."""
        idx_t = self._as_long_idx(idx, len(self))
        return torch.index_select(self.y, 0, idx_t)

    def __len__(self): return self.n

    def __getitem__(self, idx: int) -> HeteroData:
        g1, g2 = self.ch1[idx], self.ch2[idx]
        y = self.y[idx].view(1, -1)  # [1,1]

        hd = HeteroData()
        hd['id'] = self.ids[idx]

        # Node stores: only x
        hd[self.ch1_name].x = g1.x
        hd[self.ch1_name].resid = g1.resid
        hd[self.ch1_name].resname = g1.resname
        hd[self.ch1_name].chain_id = g1.chain_id
        hd[self.ch2_name].x = g2.x
        hd[self.ch2_name].resid = g2.resid
        hd[self.ch2_name].resname = g2.resname
        hd[self.ch2_name].chain_id = g2.chain_id

        # Edge stores: edge_index (add edge_attr later if needed)
        et1 = (self.ch1_name, "intra", self.ch1_name)
        et2 = (self.ch2_name, "intra", self.ch2_name)
        hd[et1].edge_index = g1.edge_index
        hd[et2].edge_index = g2.edge_index

        # Attention Mask between channels -> using chain_id
        if self.mask_between:
            for g, channel in zip([g1, g2], [self.ch1_name, self.ch2_name]):

                hd[channel].mask = torch.tensor(
                    [int(self.mask_chain_map.get(str(cid), 0)) for cid in g.chain_id],
                    dtype=torch.bool
                )

        # Graph-level label
        hd["y"] = y

        return hd