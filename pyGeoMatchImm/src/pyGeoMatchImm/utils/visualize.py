import __main__
__main__.pymol_argv = [ 'pymol', '-qi' ]
import pymol
from pymol import cmd
import numpy as np
from umap import UMAP
import networkx as nx
import matplotlib.pyplot as plt
import logging as log
import os
import logging

log = logging.getLogger(__name__)

def print_pymol_resids(input: dict):
    for channel_name, channel_data in input.items():
        print(f"Channel: {channel_name}")
        resids = channel_data.get('resids', [])
        if resids:
            resids_str = " ".join(f"(resid {r} and chain {c})" for r,c in resids)
            print(resids_str)
        else:
            print("  No residues found.")

def launch_pymol(input: dict):
    # Call the function below before using any PyMOL modules.
    pymol.finish_launching()  # not supported on macOS

    log.info("Loading structures in PyMOL...")
    log.info(input)
    
    for channel_name, channel_data in input.items():
        cmd.load(channel_data.get('path'), object=channel_name, discrete=1)

        resids = channel_data.get('resids', [])
        if resids:
            resids = " ".join(f"(resid {r} and chain {c})" for r,c in resids)
        
        cmd.show_as("sticks", resids)
        cmd.set('stereo_shift', 0.23)
        cmd.set('stereo_angle', 1.0)


def save_graph(G: nx.Graph, channel_name: str):
    mhc_res_list = []
    pep_res_list = []
    for nodes in G.nodes:
        if nodes.startswith('A'):
            G.nodes[nodes]['label'] = "MHC:" + nodes.split(':')[2]
            G.nodes[nodes]['chain_id'] = 'blue'
            mhc_res_list.append(nodes.split(':')[2])

        elif nodes.startswith('C'):
            G.nodes[nodes]['label'] = "p:"  + nodes.split(':')[2]
            G.nodes[nodes]['chain_id'] = 'red'
            pep_res_list.append(nodes.split(':')[2])
        elif nodes.startswith('D'):
            G.nodes[nodes]['label'] = "a:"  + nodes.split(':')[2]
            G.nodes[nodes]['chain_id'] = 'green'
            pep_res_list.append(nodes.split(':')[2])
        elif nodes.startswith('E'):
            G.nodes[nodes]['label'] = "b:"  + nodes.split(':')[2]
            G.nodes[nodes]['chain_id'] = 'yellow'
            pep_res_list.append(nodes.split(':')[2])
        else:
            G.nodes[nodes]['label'] = None
            G.nodes[nodes]['chain_id'] = None

    node_colors = [G.nodes[node]['chain_id'] for node in G.nodes]
    labels = {node: data['label'] for node, data in G.nodes(data=True)}


    edge_colors = [
        "blue"   if u.startswith("A") and v.startswith("A") else
        "red"    if u.startswith("C") and v.startswith("C") else
        "green"  if u.startswith("D") and v.startswith("D") else
        "yellow" if u.startswith("E") and v.startswith("E") else
        "gray"
        for u, v in G.edges
    ]
    
    # Get a circular layout based on the ordered nodes
    plt.figure(figsize=(6.5, 6.5))
    plt.margins(0.03, 0.03)
    plt.box(False)

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=400, node_color=node_colors, edgecolors="tab:gray")
    nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="black", font_weight="bold")
    plt.show()
    #plt.savefig(f'{channel_name}.png', bbox_inches='tight', dpi=400)


def visualize_embeddings(xc, y=None, epoch=None, save_dir='.', suffix ='', ids = list[str]):

    xc_np = [x.detach().cpu().numpy() for x in xc]
    xc_np = np.vstack(xc_np)
    if y is not None:
        y_np = [x.detach().cpu().numpy() for x in y]
    else:
        y_np = np.zeros(len(xc_np))

    #check number of samples
    if xc_np.shape[0] < 10:
        log.warning("No embeddings to visualize.")
        return
    
    reducer = UMAP(n_neighbors=5, min_dist=0.01, random_state=42, metric='cosine')
    X_umap = reducer.fit_transform(xc_np)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                          c=y_np, cmap='viridis', alpha=0.8,
                          vmin=0, vmax=1 )
    plt.colorbar(scatter, label="Label (y)", )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    if ids is not None:
        for i in range(X_umap.shape[0]):
            label_id = ids[i]
            plt.annotate(str(label_id),
                          (X_umap[i, 0], X_umap[i, 1]),
                          textcoords="offset points",
                        xytext=(3, 3),
                        ha='left',
                        fontsize=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))

    plt.tight_layout()

    if epoch is not None:
        os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
        plt.savefig(f"{save_dir}/embeddings/umap_epoch_{epoch}_{suffix}.png", bbox_inches='tight', dpi=400)

    plt.close()

def correlation_pred_labels(y_true, y_pred, epoch=None, save_dir='.', suffix=''):

    y_true_np = [x.detach().cpu().numpy() for x in y_true]
    y_pred_np = [x.detach().cpu().numpy() for x in y_pred]

    plt.figure(figsize=(6, 5))
    y_true_np = np.array(y_true_np).ravel()
    y_pred_np = np.array(y_pred_np).ravel()
    colors = ['tab:blue' if y == 0 else 'tab:orange' for y in y_true_np]

    corr_coef = np.corrcoef(y_true_np, y_pred_np)[0, 1]
    lm_line = np.polyfit(y_true_np, y_pred_np, 1)

    plt.scatter(y_true_np, y_pred_np, alpha=0.7, c=colors, edgecolors='k', linewidths=0.4)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.plot([0, 1], [lm_line[1], lm_line[0] + lm_line[1]], 'g-')  # Linear fit line
    plt.text(0.05, 0.9, f'Corr Coef: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Correlation between True and Predicted Labels")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()

    if epoch is not None:
        os.makedirs(f"{save_dir}/correlation", exist_ok=True)
        plt.savefig(f"{save_dir}/correlation/correlation_epoch_{epoch}.png", bbox_inches='tight', dpi=400)

    plt.close()