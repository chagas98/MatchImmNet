#!/usr/bin/env python3

from ..utils.base import TO_3LETTERS 
from huggingface_hub import login
from esm.models.esm3 import ESM3, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.structure.protein_chain import ProteinChain
import networkx as nx

import torch

import os
import logging

log = logging.getLogger(__name__)

# Disable tokenizer threads to avoid fork warning
#torch.multiprocessing.set_start_method("spawn", force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ESM3Embedder:
    def __init__(self, model_name=ESM3_OPEN_SMALL, device="cuda", hf_token="hf.key", concat_embed=None, **kwargs):

        self.concat = concat_embed  # concatenate seqs with | (sequence_tokenizer.py in esm3)
        self.model_name = model_name
        self.device = device

        if "HF_TOKEN" not in os.environ:
            # Hugging Face settings
            os.environ["HF_TOKEN"] = open(hf_token).read().strip()
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                log.info("Successfully logged in to Hugging Face!")
            else:
                raise ValueError("Hugging Face token not found in environment variable 'HF_TOKEN'.")

    def init_model(self):
        if not hasattr(self, 'model'):
            self.model = ESM3.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        return self.model

    def __call__(self, structures: dict = None, graphs: dict = None) -> torch.Tensor:
        
        self.structures = structures
        self.graphs = graphs

        # Embed sequences and collect embeddings
        residx_embs = {}
        match self.concat:

            case "all": #embedding for all chains concatenated
                sequence, lengths, protchains = self._seq_all_concat()
                residx_embs = self._embed_sequence(sequence, lengths, protchains)

            case "partial": #embedding for each channel
                
                for _, file in self.structures.items():
                    protein_tensor, lengths, protchains = self._seq_partial_concat(file)
                    residx_embs.update(self._embed_sequence(protein_tensor, lengths, protchains))

            case "chain": #embedding for each chain separately
                for _, file in self.structures.items():
                    protein_complex = ProteinComplex.from_pdb(file)
                    protchains = {protchain.chain_id: protchain for protchain in protein_complex.chain_iter()}
                    for chain in protchains.keys():
                        protein = ESMProtein.from_pdb(file, chain_id=chain)  # validate PDB file
                        seq = protein.sequence

                        residx_embs.update(self._embed_sequence(seq, {chain: (0, len(seq))}, protchains))
            case _:
                raise ValueError(f"Unknown concat method: {self.concat}")


        return self._collect_node_embeddings(residx_embs)

    def _seq_all_concat(self):
        
        lengths = {}
        previous = 0
        all_sequences = []
        protchains = {}
        for _, file in self.structures.items():
            protein_complex = ProteinComplex.from_pdb(file)

            protchains.update({protchain.chain_id: protchain for protchain in protein_complex.chain_iter()})

        protchains = dict(sorted(protchains.items(), key=lambda x: x[0]))  # sort by chain_id, keep as dict

        for chain_id, seq in protchains.items():
            lengths[chain_id] = (previous, previous + len(seq.sequence))
            previous += len(seq.sequence) + 1  # +1 for separator
            all_sequences.append(seq.sequence)

        protein_sequence = "|".join(all_sequences)
        
        for name, (start, end) in lengths.items():
            log.debug(f"{name}: {start}-{end} (len={end-start})")
            log.debug(f"  seq: {protein_sequence[start:end]}")

        return protein_sequence, lengths, protchains

    def _seq_partial_concat(self, file):

        protein_complex = ProteinComplex.from_pdb(file)

        protchains = {protchain.chain_id: protchain for protchain in protein_complex.chain_iter()}
        
        lengths = {}
        previous = 0
        all_sequences = []

        for chain_id, seq in protchains.items():
            lengths[chain_id] = (previous, previous + len(seq.sequence))
            previous += len(seq.sequence) + 1  # +1 for separator

        protein_tensor = ESMProtein.from_protein_complex(protein_complex)

        return protein_tensor, lengths, protchains
        
    
    def _embed_sequence(self, protein: ESMProtein | str, lengths: dict, protchains: dict[str, ProteinChain]) -> bool:

        if isinstance(protein, str):
            protein = ESMProtein(sequence=protein)

        with torch.no_grad():
            protein_tensor = self.model.encode(protein).to("cpu")
            output = self.model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
            protein_embeddings = output.per_residue_embedding[1:-1].detach().to("cpu")  # drop <cls> and <eos>
            cls = output.per_residue_embedding[0].detach().to("cpu")  # <cls> token

        del output, protein_tensor
        torch.cuda.empty_cache()
        
        residues_embs = {}
        for chain, (start, end) in lengths.items():

            emb = protein_embeddings[start:end]

            # metadata from embeddings
            protchain = protchains[chain]
            res_idx = protchain.residue_index
            sequence = list(protchain.sequence)

            assert len(sequence) == (end - start), f"Length mismatch for chain {chain}"
            assert chain == protchain.chain_id, f"Chain ID mismatch: {chain} != {protchain.chain_id}"
            assert len(res_idx) == len(sequence), f"Residue index and sequence length mismatch for chain {chain}"
            assert emb.shape[0] == len(sequence), f"Embedding length mismatch for sequence chain {chain}"

            for i, (res, idx) in enumerate(zip(sequence, res_idx)):
                
                res3 = TO_3LETTERS.get(res, "UNK")
                
                if res3 == "UNK":
                    raise ValueError(f"Unknown residue {res} in chain {chain} at index {idx}")
                
                residues_embs[f"{chain}:{res3}:{idx}"] = emb[i]
        
        # Free up GPU memory
        torch.cuda.empty_cache()
        del protein_embeddings

        # Check current and peak GPU memory usage
        log.debug(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        log.debug(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        # Optionally reset peak stats
        torch.cuda.reset_peak_memory_stats()

        # After your inference or training step:
        log.debug(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        return residues_embs
        
    def _collect_node_embeddings(self, residx_embs: dict):

        graphs_updated = self.graphs.copy()

        for channel, graph in graphs_updated.items():
            node_attr_map = {}

            for node_id, node_data in graph.nodes(data=True):
                # Standardize node key
                if len(node_id.split(':')) != 3:
                    node_key = f"{node_data['chain_id']}:{node_data['residue_name']}:{node_data['residue_number']}"
                else:
                    node_key = node_id

                if node_key not in residx_embs:
                    raise ValueError(f"Node {node_key} not found in ESM embeddings")

                node_attr_map[node_id] = residx_embs[node_key]

            # assign per-node embeddings
            nx.set_node_attributes(graph, node_attr_map, "esm3")
            
        return graphs_updated



#https://medium.com/stanford-cs224w/protein-chemical-interaction-prediction-using-graph-neural-networks-f063de1df7e0
#https://github.com/evolutionaryscale/esm
