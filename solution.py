# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 9: Tracking-by-detection with an integer linear program (ILP)
#
# You could also run this notebook on your laptop, a GPU is not needed :).
#
# <div class="alert alert-danger">
# Set your python kernel to <code>09-tracking</code>
# </div>
#
# You will learn:
# - how to represent tracking inputs and outputs as a graph using the `networkx` library
# - how to use [`motile`](https://funkelab.github.io/motile/) to solve tracking via global optimization
# - how to visualize tracking inputs and outputs
# - how to evaluate tracking and understand common tracking metrics
# - how to add custom costs to the candidate graph and incorpate them into `motile`
# - how to learn the best **hyperparameters** of the ILP using an SSVM (bonus)
#
#
# Places where you are expected to write code are marked with
# ```
# ### YOUR CODE HERE ###
# ```
#
# This notebook was originally written by Benjamin Gallusser, and was edited for 2024 by Caroline Malin-Mayor.

# %% [markdown]
# ## Import packages

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Notebook at full width in the browser
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

import time
from pathlib import Path

import skimage
import pandas as pd
import numpy as np
import napari
import networkx as nx
import plotly.io as pio
import scipy

pio.renderers.default = "vscode"

import motile
from motile.plot import draw_track_graph, draw_solution
from utils import InOutSymmetry, MinTrackLength

import traccuracy
from traccuracy import run_metrics
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import CTCMatcher
import zarr
from motile_toolbox.visualization import to_napari_tracks_layer
from napari.layers import Tracks
from csv import DictReader

from tqdm.auto import tqdm

from typing import Iterable, Any

# %% [markdown]
# ## Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt. The raw data, pre-computed segmentations, and detection probabilities are saved in a zarr, and the ground truth tracks are saved in a csv. The segmentation was generated with a pre-trained StartDist model, so there may be some segmentation errors which can affect the tracking process. The detection probabilities also come from StarDist, and are downsampled in x and y by 2 compared to the detections and raw data.

# %%
data_path = "data/breast_cancer_fluo.zarr"
data_root = zarr.open(data_path, 'r')
image_data = data_root["raw"][:]
segmentation = data_root["seg_relabeled"][:]
probabilities = data_root["probs"][:]


# %% [markdown]
# ## Task 1: Read in the ground truth graph
# The ground truth tracks are stored in a CSV with five columns: id, time, x, y, and parent_id.
#
# Each row in the CSV represents a detection at location (time, x, y) with the given id.
# If the parent_id is not -1, it represents the id of the parent in the previous time frame.
# For cell tracking, tracks can usually be stored in this format, because there is no merging.
# With merging, a more complicated data struture would be needed.
#
# <div class="alert alert-block alert-info"><h3>Task 1: Read in the ground truth graph</h3>
#
# For this task, you will read in the csv and store the tracks as a `networkx` DiGraph.
# Each node in the graph will represent a detection, and should use the given id, and have attributes `time` and `pos` to represent time and position (a list of [x, y]).
# Each edge in the graph will go from a parent to a child.
# </div>

# %%
def read_gt_tracks():
    gt_tracks = nx.DiGraph()
    ### YOUR CODE HERE ###
    return gt_tracks

gt_tracks = read_gt_tracks()


# %% tags=["solution"]
def read_gt_tracks():
    with open("data/breast_cancer_fluo_gt_tracks.csv") as f:
        reader = DictReader(f)
        gt_tracks = nx.DiGraph()
        for row in reader:
            _id = int(row["id"])
            attrs = {
                "pos": [float(row["x"]), float(row["y"])],
                "time": int(row["time"]),
            }
            parent_id = int(row["parent_id"])
            gt_tracks.add_node(_id, **attrs)
            if parent_id != -1:
                gt_tracks.add_edge(parent_id, _id)
    return gt_tracks

gt_tracks = read_gt_tracks()

# %% [markdown]
# Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) to visualize the data. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks. If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html).

# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(image_data, name="raw")
viewer.add_labels(segmentation, name="seg")
viewer.add_image(probabilities, name="probs", scale=(1, 2, 2))
tracks_layer = to_napari_tracks_layer(gt_tracks, frame_key="time", location_key="pos", name="gt_tracks")
viewer.add_layer(tracks_layer)

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ## Task 2: Build a candidate graph from the detections
#
# <div class="alert alert-block alert-info"><h3>Task 2: Build a candidate graph</h3>
# </div>

# %% [markdown]
# We will represent a linking problem as a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) that contains all possible detections (graph nodes) and links (graph edges) between them.
#
# Then we remove certain nodes and edges using discrete optimization techniques such as an integer linear program (ILP).
#
# First of all, we will build a candidate graph built from the detected cells in the video.


# %%
gt_trackgraph = motile.TrackGraph(gt_tracks, frame_attribute="time")

def nodes_from_segmentation(
    segmentation: np.ndarray, probabilities: np.ndarray
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x), where h is the number of hypotheses.
        probabilities (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x), where h is the number of hypotheses.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        segs = segmentation[t]
        nodes_in_frame = []
        props = skimage.measure.regionprops(segs)
        for regionprop in props:
            node_id = regionprop.label
            attrs = {
                "time": t,
            }
            attrs["label"] = regionprop.label
            centroid = regionprop.centroid  #  y, x
            attrs["pos"] = centroid
            probability = probabilities[t, int(centroid[0] // 2), int(centroid[1] // 2)]
            attrs["prob"] = probability
            assert node_id not in cand_graph.nodes
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].extend(nodes_in_frame)
    return cand_graph, node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [cand_graph.nodes[node]["pos"] for node in node_ids]
    return scipy.spatial.KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree

cand_graph, node_frame_dict = nodes_from_segmentation(segmentation, probabilities)
print(cand_graph.number_of_nodes())
add_cand_edges(cand_graph, max_edge_distance=50, node_frame_dict=node_frame_dict)
cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="time")


# %% [markdown]
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have visualized our data in napari and set up a candidate graph with all possible detections and links that we could select with our optimization task. </h3>
#
# We will now together go through the `motile` <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example before you actually set up and run your own motile optimization.
# </div>

# %% [markdown]
# ## Setting Up the Tracking Optimization Problem

# %% [markdown]
# As hinted earlier, our goal is to prune the candidate graph. More formally we want to find a graph $\tilde{G}=(\tilde{V}, \tilde{E})$ whose vertices $\tilde{V}$ are a subset of the candidate graph vertices $V$ and whose edges $\tilde{E}$ are a subset of the candidate graph edges $E$.
#
#
# Finding a good subgraph $\tilde{G}=(\tilde{V}, \tilde{E})$ can be formulated as an [integer linear program (ILP)](https://en.wikipedia.org/wiki/Integer_programming) (also, refer to the tracking lecture slides), where we assign a binary variable $x$ and a cost $c$ to each vertex and edge in $G$, and then computing $min_x c^Tx$.
#
# A set of linear constraints ensures that the solution will be a feasible cell tracking graph. For example, if an edge is part of $\tilde{G}$, both its incident nodes have to be part of $\tilde{G}$ as well.
#
# `motile` ([docs here](https://funkelab.github.io/motile/)), makes it easy to link with an ILP in python by implementing commong linking constraints and costs. 

# %% [markdown]
# ## Task 3 - Basic Tracking with Motile
# <div class="alert alert-block alert-info"><h3>Task 3: Set up a basic motile tracking pipeline</h3>
# <p>Use the motile <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example to set up a basic motile pipeline for our task. Then run the function and find hyperparmeters that give you tracks.</p>
# </div>
#

# %%
def solve_basic_optimization(graph, edge_weight, edge_constant):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.
        edge_weight (float): The weighting factor of the edge selection cost.
        edge_constant(float): The constant cost of selecting any edge.

    Returns:
        motile.Solver: The solver object, ready to be inspected.
    """
    solver = motile.Solver(graph)
    ### YOUR CODE HERE ###
    solution = solver.solve()

    return solver


# %% tags=["solution"]
def solve_basic_optimization(graph, edge_weight, edge_constant):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.
        edge_weight (float): The weighting factor of the edge selection cost.
        edge_constant(float): The constant cost of selecting any edge.

    Returns:
        motile.Solver: The solver object, ready to be inspected.
    """
    solver = motile.Solver(graph)

    solver.add_costs(
        motile.costs.EdgeDistance(weight=edge_weight, constant=edge_constant, position_attribute="pos")
    )

    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))

    solution = solver.solve()

    return solver


# %% [markdown]
# Here is a utility function to gauge some statistics of a solution.

# %%
from motile_toolbox.candidate_graph import graph_to_nx
def print_solution_stats(solver, graph, gt_graph):
    """Prints the number of nodes and edges for candidate, ground truth graph, and solution graph.

    Args:
        solver: motile.Solver, after calling solver.solve()
        graph: motile.TrackGraph, candidate graph
        gt_graph: motile.TrackGraph, ground truth graph
    """
    time.sleep(0.1)  # to wait for ilpy prints
    print(
        f"\nCandidate graph\t\t{len(graph.nodes):3} nodes\t{len(graph.edges):3} edges"
    )
    print(
        f"Ground truth graph\t{len(gt_graph.nodes):3} nodes\t{len(gt_graph.edges):3} edges"
    )
    solution = graph_to_nx(solver.get_selected_subgraph())

    print(f"Solution graph\t\t{solution.number_of_nodes()} nodes\t{solution.number_of_edges()} edges")


# %% [markdown]
# Here we actually run the optimization, and compare the found solution to the ground truth.
#
# <div class="alert alert-block alert-warning"><h3>Gurobi license error</h3>
# Please ignore the warning `Could not create Gurobi backend ...`.
#
#
# Our integer linear program (ILP) tries to use the proprietary solver Gurobi. You probably don't have a license, in which case the ILP will fall back to the open source solver SCIP.
# </div>

# %% tags=["solution"]
# Solution

edge_weight = 1
edge_constant=-20
solver = solve_basic_optimization(cand_trackgraph, edge_weight, edge_constant)
solution_graph = graph_to_nx(solver.get_selected_subgraph())
print_solution_stats(solver, cand_trackgraph, gt_trackgraph)

"""
Explanation: Since the ILP formulation is a minimization problem, the total weight of each node and edge needs to be negative.
The cost of each node corresponds to its detection probability, so we can simply mulitply with `node_weight=-1`.
The cost of each edge corresponds to 1 - distance between the two nodes, so agai we can simply mulitply with `edge_weight=-1`.

Futhermore, each detection (node) should maximally be linked to one other detection in the previous and next frames, so we set `max_flow=1`.
"""


# %% [markdown]
# ## Visualize the Result

# %%
tracks_layer = to_napari_tracks_layer(solution_graph, frame_key="time", location_key="pos", name="solution_tracks")
viewer.add_layer(tracks_layer)

# %% [markdown]
# ### Recolor detections in napari according to solution and compare to ground truth


# %%
def relabel_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results so that nodes in same
    track share the same id. IDs do change at division.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for relabeling. Nodes not in graph will be removed from seg. Original
            segmentation ids and hypothesis ids have to be stored in the graph so we
            can map them back.
        segmentation (np.ndarray): Original (potentially multi-hypothesis)
            segmentation with dimensions (t,h,[z],y,x), where h is 1 for single
            input segmentation.

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,1,[z],y,x)
    """
    tracked_masks = np.zeros_like(segmentation)
    id_counter = 1
    parent_nodes = [n for (n, d) in solution_nx_graph.out_degree() if d > 1]
    soln_copy = solution_nx_graph.copy()
    for parent_node in parent_nodes:
        out_edges = solution_nx_graph.out_edges(parent_node)
        soln_copy.remove_edges_from(out_edges)
    for node_set in nx.weakly_connected_components(soln_copy):
        for node in node_set:
            time_frame = solution_nx_graph.nodes[node]["time"]
            previous_seg_id = solution_nx_graph.nodes[node]["label"]
            previous_seg_mask = (
                segmentation[time_frame] == previous_seg_id
            )
            tracked_masks[time_frame][previous_seg_mask] = id_counter
        id_counter += 1
    return tracked_masks


solution_seg = relabel_segmentation(solution_graph, segmentation)
viewer.add_labels(solution_seg, name="solution_seg")

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ## Evaluation Metrics
#
# We were able to understand via visualizing the predicted tracks on the images that the basic solution is far from perfect for this problem.
#
# Additionally, we would also like to quantify this. We will use the package [`traccuracy`](https://traccuracy.readthedocs.io/en/latest/) to calculate some [standard metrics for cell tracking](http://celltrackingchallenge.net/evaluation-methodology/). For example, a high-level indicator for tracking performance is called TRA.
#
# If you're interested in more detailed metrics, you can check out for example the false positive (FP) and false negative (FN) nodes, edges and division events.


# %%
def get_metrics(gt_graph, labels, pred_graph, pred_segmentation):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.

    Returns:
        results (dict): Dictionary of metric results.
    """

    gt_graph = traccuracy.TrackingGraph(
        graph=gt_graph,
        frame_key="time",
        label_key="show",
        location_keys=("x", "y"),
        segmentation=labels,
    )

    pred_graph = traccuracy.TrackingGraph(
        graph=pred_graph,
        frame_key="time",
        label_key="show",
        location_keys=("x", "y"),
        segmentation=pred_segmentation,
    )

    results = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )

    return results


# %%
get_metrics(gt_nx_graph, None, solution_graph, solution_seg)


# %% [markdown]
# ## Task 4 - Add an appear cost, but not at the boundary
# The [Appear](https://funkelab.github.io/motile/api.html#motile.costs.Appear_) cost penalizes starting a new track, encouraging continuous tracks. However, you do not want to penalize tracks that appear in the first frame. In our case, we probably also do not want to penalize appearing at the "bottom" of the dataset. The built in Appear cost has an `ignore_attribute` argument, where if the node has that attribute and it evaluates to True, the Appear cost will not be paid for that node.
#
# <div class="alert alert-block alert-info"><h3>Task 4: Add an appear cost, but not at the boundary</h3>
# <p> Add an attribute to the nodes of our candidate graph that is True if the appear cost should NOT be paid for that node, and False (or not present) otherwise. Then add an Appear cost to our motile pipeline using our new attribute as the `ignore_attribute` argument, and re-solve to see if performance improves.</p>
# </div>

# %%
def add_appear_ignore_attr(cand_graph):
    ### YOUR CODE HERE ###
    pass  # delete this

add_appear_ignore_attr(cand_graph)


# %% tags=["solution"]
def add_appear_ignore_attr(cand_graph):
    for node in cand_graph.nodes():
        time = cand_graph.nodes[node]["time"]
        pos_x = cand_graph.nodes[node]["pos"][0]
        if time == 0 or pos_x >= 710:
            cand_graph.nodes[node]["ignore_appear"] = True

add_appear_ignore_attr(cand_graph)
cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="time")


# %%
def solve_appear_optimization(graph, edge_weight, edge_constant):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.
        edge_weight (float): The weighting factor of the edge selection cost.
        edge_constant(float): The constant cost of selecting any edge.

    Returns:
        motile.Solver: The solver object, ready to be inspected.
    """
    solver = motile.Solver(graph)

    solver.add_costs(
        motile.costs.EdgeDistance(weight=edge_weight, constant=edge_constant, position_attribute="pos")
    )
    solver.add_costs(
        motile.costs.Appear(constant=50, ignore_attribute="ignore_appear") 
    )

    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))

    solution = solver.solve()

    return solver

solver = solve_appear_optimization(cand_trackgraph, 1, -20)
solution_graph = graph_to_nx(solver.get_selected_subgraph())

# %%
tracks_layer = to_napari_tracks_layer(solution_graph, frame_key="time", location_key="pos", name="solution_appear_tracks")
viewer.add_layer(tracks_layer)
solution_seg = relabel_segmentation(solution_graph, segmentation)
viewer.add_labels(solution_seg, name="solution_appear_seg")

# %%
get_metrics(gt_tracks, None, solution_graph, solution_seg)

# %% [markdown]
# ## Checkpoint 2
# <div class="alert alert-block alert-success"><h3>Checkpoint 2</h3>
# We have run an ILP to get tracks, visualized the output, evaluated the results, and added an Appear cost that does not take effect at the boundary. If you reach this Checkpoint early, try adjusting your weights or using different combinations of Costs and Constraints to get better results. For now, stick to those implemented in motile, but consider what kinds of custom costs and constraints you could implement to improve performance, since that is what we will do next!
#
# When most people have reached this checkpoint, we will go around and
# share what worked and what didn't, and discuss ideas for custom costs or constraints.
# </div>

# %% [markdown]
# ## Customizing the Tracking Task
#
# There 3 main ways to encode prior knowledge about your task into the motile tracking pipeline.
# 1. Add an attribute to the candidate graph and incorporate it with a Selection cost
# 2. Change the structure of the candidate graph
# 3. Add a new type of cost or constraint

# %% [markdown]
# # Task 5 - Incorporating Known Direction of Motion
#
# Motile has built in the EdgeDistance as an edge selection cost, which penalizes longer edges by computing the Euclidean distance between the endpoints. However, in our dataset we see a trend of upward motion in the cells, and the false detections at the top are not moving. If we penalize movement based on what we expect, rather than Euclidean distance, we can select more correct cells and penalize the non-moving artefacts at the same time.
#  
# <div class="alert alert-block alert-info"><h3>Task 5: Incorporating known direction of motion</h3>
# <p> For this task, we need to determine the "expected" amount of motion, then add an attribute to our candidate edges that represents distance from the expected motion direction. Finally, we can incorporate that feature into the ILP via the EdgeSelection cost and see if it improves performance.</p>
# </div>

# %%
drift = ... ### YOUR CODE HERE ###

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        ### YOUR CODE HERE ###
        # get the location of the endpoints of the edge
        # then compute the distance between the expected movement and the actual movement
        # and save it in the "drift_dist" attribute (below)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)
cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="time")

# %% tags=["solution"]
drift = np.array([-20, 0])

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        source, target = edge
        source_pos = np.array(cand_graph.nodes[source]["pos"])
        target_pos = np.array(cand_graph.nodes[target]["pos"])
        expected_target_pos = source_pos + drift
        drift_dist = np.linalg.norm(expected_target_pos - target_pos)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)
cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="time")


# %%
def solve_drift_optimization(graph, edge_weight, edge_constant):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.
        edge_weight (float): The weighting factor of the edge selection cost.
        edge_constant(float): The constant cost of selecting any edge.

    Returns:
        motile.Solver: The solver object, ready to be inspected.
    """
    solver = motile.Solver(graph)

    solver.add_costs(
        motile.costs.EdgeSelection(weight=edge_weight, constant=edge_constant, attribute="drift_dist")
    )
    solver.add_costs(
        motile.costs.Appear(constant=50, ignore_attribute="ignore_appear") 
    )

    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))

    solution = solver.solve()

    return solver

solver = solve_drift_optimization(cand_trackgraph, 1, -20)
solution_graph = graph_to_nx(solver.get_selected_subgraph())

# %%
tracks_layer = to_napari_tracks_layer(solution_graph, frame_key="time", location_key="pos", name="solution_tracks_with_drift")
viewer.add_layer(tracks_layer)

solution_seg = relabel_segmentation(solution_graph, segmentation)
viewer.add_labels(solution_seg, name="solution_seg_with_drift")

# %%
get_metrics(gt_nx_graph, None, solution_graph, solution_seg)

# %% [markdown]
# ## Bonus: Learning the Weights

# %% [markdown]
#
