#! /usr/bin/env python3

# local toolkit
from ...utils.registry import register
from ...utils.base import TrainConfigs, SamplePair, _PAIRS_ANNOTATION, PairsAnnotation

# third-parties
import torch
from torch_geometric.data import Data
from typing import Callable, Dict
import graphein.protein as gp
from functools import partial
import networkx as nx
from graphein.protein.graphs import construct_graph
from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphDataset
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.subgraphs import extract_surface_subgraph, extract_subgraph, extract_subgraph_from_chains, extract_subgraph_by_sequence_position
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions)
import logging
log = logging.getLogger(__name__)

@register("graph.graphein")
class GrapheinGeneratorRes:

    _EDGE_FUNCTIONS: Dict[str, Callable] = {
        "peptide_bonds"                 : add_peptide_bonds,
        "hydrogen_bonds"                : add_hydrogen_bond_interactions,
        "disulfide_bonds"               : add_disulfide_interactions,
        "ionic_interactions"            : add_ionic_interactions,
        "aromatic_interactions"         : add_aromatic_interactions,
        "aromatic_sulphur_interactions" : add_aromatic_sulphur_interactions,
        "cation_pi_interactions"        : add_cation_pi_interactions,
        "distance_threshold"            : partial(gp.add_distance_threshold, long_interaction_threshold=1)
    }

    _NODE_FUNCTIONS: Dict[str, Callable] = {
        "amino_acid_one_hot"     : gp.amino_acid_one_hot,
        "hbond_acceptors"        : partial(gp.hydrogen_bond_acceptor, sum_features=False),
        "hbond_donors"           : partial(gp.hydrogen_bond_donor, sum_features=False),
        "meiler_embedding"       : gp.meiler_embedding,
        "expasy_protein_scale"   : partial(gp.expasy_protein_scale, add_separate=True),
        "dssp_config"            : gp.DSSPConfig()
    }

    _GRAPH_FUNCTIONS: Dict[str, Callable] = {
        "esm"                  : gp.esm_sequence_embedding,
        "rsa"                 : gp.rsa,
        "secondary_structure" : gp.secondary_structure
    }

    def __init__(self, edge_params: list, node_params: list, graph_params: list, 
                 other_params: dict = {}, dist_threshold: float = 8.0, **kwargs):
        self.edge_params  = edge_params or ['distance_threshold']
        self.node_params  = node_params or ['amino_acid_one_hot']
        self.graph_params = graph_params
        self.other_params = other_params
        self.rsa_threshold = other_params.get("rsa_threshold", .2)
        self.dist_threshold = dist_threshold

    def __call__(self, sample: SamplePair) -> SamplePair:

        self.sample = sample
        
        self.graph_config = gp.ProteinGraphConfig(
            **{ **self.other_params,
                **self._edges_functions(),
                **self._nodes_functions(),
                **self._graph_functions()
            })
        
        return self._graph_constructor()
    
    def _edges_functions(self):
       if 'distance_threshold' in self.edge_params:
           dist_func = self._EDGE_FUNCTIONS['distance_threshold']
           self._EDGE_FUNCTIONS['distance_threshold'] = partial(dist_func, threshold=self.dist_threshold)
       return {"edge_construction_functions": list(self._EDGE_FUNCTIONS[e] for e in self.edge_params if e in self._EDGE_FUNCTIONS)}

    def _nodes_functions(self):

        _node_function = {"node_metadata_functions": []}
        for n in self.node_params:
            
            if n == "dssp_config":
                _node_function["dssp_config"] = self._NODE_FUNCTIONS.get(n, None)
            else:
                _node_function["node_metadata_functions"].append(self._NODE_FUNCTIONS.get(n, None))

        return _node_function
    
    def _graph_functions(self):

        if self.graph_params == []:
            return {"graph_metadata_functions": []}
        
        return {"graph_metadata_functions": list(self._GRAPH_FUNCTIONS[g] for g in self.graph_params if g in self._GRAPH_FUNCTIONS)}

    def _graph_constructor(self):
        
        graphs = {}
        nodes = []
        edges = []  
        for sstruct in self.sample.struct:
            pdb_path = sstruct.filepath

            log.debug(f'CHANNEL NAME: {sstruct.channel}')
            if pdb_path:
                g = construct_graph(config=self.graph_config, path=pdb_path)

            else:
                log.critical('This channel does not have a PDB path:')
                raise ValueError("PDB path not found.")
            
            str_infos = sstruct.str_info
            keep_nodes = []
            for chain, residues in str_infos.items():

                if chain not in g.graph['chain_ids']:
                    raise ValueError(f"Chain '{chain}' not found in the graph.")

                log.debug(f'Extracting nodes for chain: {chain}')

                # Extract subgraph for the chain
                g_chain = extract_subgraph_from_chains(g, chains=[chain], return_node_list=False)

                # Extract exposed nodes
                if PairsAnnotation.is_chain_rsa_enabled(chain):
                    log.debug(f'Extracting surface nodes for chain: {chain} with RSA threshold {self.rsa_threshold}')
                    log.debug(f'_____RESIDUES: {residues}')
                    nodes_selected = extract_subgraph_by_sequence_position(
                        g_chain, sequence_positions=residues, return_node_list=False
                    )

                    nodes_selected = extract_surface_subgraph(
                        nodes_selected,
                        rsa_threshold=self.rsa_threshold,
                        return_node_list=True
                    )
                else:
                    nodes_selected = extract_subgraph_by_sequence_position(
                        g_chain, sequence_positions=residues, return_node_list=True
                    )
    
                keep_nodes.extend(nodes_selected)

            log.debug(f'____KEEP NODES: {keep_nodes}\n')
            graph = extract_subgraph(g, node_list=keep_nodes, return_node_list=False)

            # Processing nodes and edges
            for node in graph.nodes(data=True):
                resid = node[0].split(':')
                nodes.append(tuple((resid[2], resid[0])))

            for edge in graph.edges(data=True):
                edges.append((edge[0], edge[1]))

            #sstruct.graph = {
            #    'graph': graph,
            #    'nodes': nodes,
            #    'edges': edges
            #}
            sstruct.graph = graph

            log.debug(f'UPDATED SAMPLE: {self.sample}')

        return self.sample
    

@register("channels.graphein")
class GrapheinToChannels:
    def __init__(self, config: TrainConfigs, embed_method: str = "amino_acid_one_hot"):

        self.cfg = config
        self.embed_method = embed_method

        # Keep track of which columns to convert
        base_columns = ["coords", "distance", "edge_index", "chain_id", "residue_number", "residue_name"] #TODO we can use this stage to get mask matrix based on structural info
        node_columns = self.cfg.node_params if self.cfg.node_params else []
        if self.embed_method not in node_columns:
            node_columns.append(self.embed_method)
        edge_columns = self.cfg.edge_params if self.cfg.edge_params else []
        graph_columns = self.cfg.graph_params if self.cfg.graph_params else []
        self.columns = list(set(base_columns + node_columns + edge_columns + graph_columns))

        # Initialize the convertor
        self._convertor  = GraphFormatConvertor(src_format="nx", dst_format="pyg", verbose="gnn", 
                                                columns = self.columns)
        self.all_dataset = []   
        self.ids         = []
        self._labels     = []   

    def __call__(self, sample: SamplePair):
        
        #graphs = sample.get_graphs
        #sequences = sample.get_sequences

        log.debug(f"Converting graphs for sample {sample.id}")

        pyg_data_channels = {}
        seq_data_channels = {}
        for ch_name in sample.channels:
            ch_graph = sample.getgraph(ch_name)

            pyg_data = self._convertor(ch_graph)  # -> torch_geometric.data.Data
            pyg_data.name = sample.id  # keep track of the sample ID

            pyg_data_channels[ch_name] = pyg_data

            sequences = sample.getseq(channel=ch_name)
            seq_data_channels[ch_name] = sequences

        label = torch.tensor([sample.label], dtype=torch.float) if sample.label is not None else None

        log.debug(f"Converted for {sample.id}")

        return sample.id, label, pyg_data_channels, seq_data_channels