# -*- coding: utf-8 -*-
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
# Objective:
# - Write a pipeline that takes in cell detections and links them across time to obtain lineage trees
#
# Methods/Tools:
#
# - **`networkx`**: To represent the tracking inputs and outputs as graphs. Tracking is often framed
#     as a graph optimization problem. Nodes in the graph represent detections, and edges represent links
#     across time. The "tracking" task is then framed as selecting the correct edges to link your detections.
# - **`motile`**: To set up and solve an Integer Linear Program (ILP) for tracking.
#     ILP-based methods frame tracking as a constrained optimization problem. The task is to select a subset of nodes/edges from a "candidate graph" of all possible nodes/edges. The subset must minimize user-defined costs (e.g. edge distance), while also satisfying a set of tracking constraints (e.g. each cell is linked to at most one cell in the previous frame). Note: this tracking approach is not inherently using
#     "deep learning" - the costs and constraints are usually hand-crafted to encode biological and data-based priors, although cost features can also be learned from data.
# - **`napari`**: To visualize tracking inputs and outputs. Qualitative analysis is crucial for tuning the
#     weights of the objective function and identifying data-specific costs and constraints.
# - **`traccuracy`**: To evaluate tracking results. Metrics such as accuracy can be misleading for tracking,
#     because rare events such as divisions are much harder than the common linking tasks, and might
#     be more biologically relevant for downstream analysis. Therefore, it is important to evaluate on
#     a wide range of error metrics and determine which are most important for your use case.
#
# After running through the full tracking pipeline, from loading to evaluation, we will learn how to **incorporate custom costs** based on dataset-specific prior information. As a bonus exercise,
# you can learn how to **learn the best cost weights** for a task from
# from a small amount of ground truth tracking information.
#
# You can run this notebook on your laptop, a GPU is not needed.
#
# <div class="alert alert-danger">
# Set your python kernel to <code>09-tracking</code>
# </div>
#
# Places where you are expected to write code are marked with
# ```
# ### YOUR CODE HERE ###
# ```
#
# This notebook was originally written by Benjamin Gallusser, and was edited for 2024 by Caroline Malin-Mayor.

# %% [markdown]
# Visualizations on a remote machine
# If you are running this notebook on a remote machine, we need to set up a few things so that you can view `napari` on the remote machine.
# 1. From VSCode connected to your remote machine, forward a port (e.g. `4000`) to your local machine.
#     - Open you command palette in VSCode (usually CMD-Shift-P) and type "forward a port"
#     - Then type in the desired port number `4000` and hit enter
#     - From the "PORTS" tab, you should see port 4000 listed as a forwarded port
# 2. Download and install [NoMachine](https://www.nomachine.com/download) on your local machine if it is not already installed.
# 3. Enter the server address in host, set the port to match the port you forwarded in step 1 and protocol as NX. Feel free to enter any name you would like.
# 4. Click on the configuration tab on the left.
# 5. Choose "Use key-based authentication with a key you provide" and hit the "Modify" button.
# 6. Provide the path to your ssh key .pem file.
# 7. Finally hit connect (or Add).
# 8. If you are asked to create a desktop, click yes.
# 9. You should then see a time and date, hitting enter should let you enter your username and access the desktop. The first login may be slow.
# 10. Still in NoMachine, open a shell window. Hit the application button in the bottom left corner and launch "Konsole"
# 11. From the shell, run `echo $DISPLAY`. Copy the output. It should be something like `:1005`
# 12. Return to your notebook in VSCode, and proceed with the exercise.
# 13. Modify the cell below to input the DISPLAY port you retrieved in step 11

# %%
import os
os.environ["DISPLAY"] = "TODO"

# %% [markdown]
# ## Import packages

# %%
import skimage
import numpy as np
import napari
import networkx as nx
import scipy

import motile

import zarr
from motile_toolbox.candidate_graph import graph_to_nx
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids
import motile_plugin.widgets as plugin_widgets
from motile_plugin.backend.motile_run import MotileRun
import traccuracy
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import IOUMatcher
from csv import DictReader

from tqdm.auto import tqdm

from typing import Iterable, Any

# %% [markdown]
# ## Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt. The raw data, pre-computed segmentations, and detection probabilities are saved in a zarr, and the ground truth tracks are saved in a csv. The segmentation was generated with a pre-trained StartDist model, so there may be some segmentation errors which can affect the tracking process. The detection probabilities also come from StarDist, and are downsampled in x and y by 2 compared to the detections and raw data.

# %% [markdown]
# Here we load the raw image data, segmentation, and probabilities from the zarr, and view them in napari.

# %%
data_path = "./data/breast_cancer_fluo.zarr"
data_root = zarr.open(data_path, 'r')
image_data = data_root["raw"][:]
segmentation = data_root["seg"][:]
probabilities = data_root["probs"][:]

# %% [markdown]
# Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) to visualize the data. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks. If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html). Here we visualize the raw data, the predicted segmentations, and the predicted probabilities as separate layers. You can toggle each layer on and off in the layers list on the left.

# %%
viewer = napari.Viewer()
viewer.add_image(probabilities, name="probs", scale=(1, 2, 2))
viewer.add_image(image_data, name="raw")
viewer.add_labels(segmentation, name="seg")

# %% [markdown]
# After running the previous cell, open NoMachine and check for an open napari window.

# %% [markdown]
# ## Read in the ground truth graph
#
# In addition to the image data and segmentations, we also have a ground truth tracking solution.
# The ground truth tracks are stored in a CSV with five columns: id, time, x, y, and parent_id.
#
# Each row in the CSV represents a detection at location (time, x, y) with the given id.
# If the parent_id is not -1, it represents the id of the parent detection in the previous time frame.
# For cell tracking, tracks can usually be stored in this format, because there is no merging.
# With merging, a more complicated data struture would be needed.
#
# Note that there are no ground truth segmentations - each detection is just a point representing the center of a cell.
#

# %% [markdown]
#
# <div class="alert alert-block alert-info"><h3>Task 1: Read in the ground truth graph</h3>
#
# For this task, you will read in the csv and store the tracks as a <a href=https://en.wikipedia.org/wiki/Directed_graph>directed graph</a> using the `networkx` library. Take a look at the documentation for the networkx DiGraph <a href=https://networkx.org/documentation/stable/reference/classes/digraph.html>here</a> to learn how to create a graph, add nodes and edges with attributes, and access those nodes and edges.
#
# Here are the requirements for the graph:
# <ol>
#     <li>Each row in the CSV becomes a node in the graph</li>
#     <li>The node id is an integer specified by the "id" column in the csv</li>
#     <li>Each node has an integer "t" attribute specified by the "time" column in the csv</li>
#     <li>Each node has float "x", "y" attributes storing the corresponding values from the csv</li>
#     <li>If the parent_id is not -1, then there is an edge in the graph from "parent_id" to "id"</li>
# </ol>
#
# You can read the CSV using basic python file io, csv.DictReader, pandas, or any other tool you are comfortable with. If not using pandas, remember to cast your read in values from strings to integers or floats.
# </div>
#

# %% tags=["task"]
def read_gt_tracks():
    gt_tracks = nx.DiGraph()
    ### YOUR CODE HERE ###
    return gt_tracks

gt_tracks = read_gt_tracks()


# %% tags=["solution"]
def read_gt_tracks():
    gt_tracks = nx.DiGraph()
    with open("data/breast_cancer_fluo_gt_tracks.csv") as f:
        reader = DictReader(f)
        for row in reader:
            _id = int(row["id"])
            attrs = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "t": int(row["time"]),
            }
            parent_id = int(row["parent_id"])
            gt_tracks.add_node(_id, **attrs)
            if parent_id != -1:
                gt_tracks.add_edge(parent_id, _id)

    return gt_tracks

gt_tracks = read_gt_tracks()

# %%
# run this cell to test your implementation
assert gt_tracks.number_of_nodes() == 5490, f"Found {gt_tracks.number_of_nodes()} nodes, expected 5490"
assert gt_tracks.number_of_edges() == 5120, f"Found {gt_tracks.number_of_edges()} edges, expected 5120"
for node, data in gt_tracks.nodes(data=True):
    assert type(node) == int, f"Node id {node} has type {type(node)}, expected 'int'"
    assert "t" in data, f"'t' attribute missing for node {node}"
    assert type(data["t"]) == int, f"'t' attribute has type {type(data['t'])}, expected 'int'"
    assert "x" in data, f"'x' attribute missing for node {node}"
    assert type(data["x"]) == float, f"'x' attribute has type {type(data['x'])}, expected 'float'"
    assert "y" in data, f"'y' attribute missing for node {node}"
    assert type(data["y"]) == float, f"'y' attribute has type {type(data['y'])}, expected 'float'"
print("Your graph passed all the tests!")

# %% [markdown]
# Here we set up a napari widget for visualizing the tracking results. This is part of the motile napari plugin, not part of core napari.
# If you get a napari error that the viewer window is closed, please re-run the previous visualization cell to re-open the viewer window.

# %%
widget = plugin_widgets.TreeWidget(viewer)
viewer.window.add_dock_widget(widget, name="Lineage View", area="right")

# %% [markdown]
# Here we add a "MotileRun" to the napari tracking visualization widget (the "view_controller"). A MotileRun includes a name, a set of tracks, and a segmentation. The tracking visualization widget will add:
# - a Points layer with the points in the tracks
# - a Tracks layer to display the track history as a "tail" behind the point in the current time frame
# - a Labels layer, if a segmentation was provided
# - a Lineage View widget, which displays an abstract graph representation of all the solution tracks
#
# These views are synchronized such that every element is colored by the track ID of the element. Clicking on a node in the Lineage View will navigate to that cell in the data, and vice versa.
#
# Hint - if your screen is too small, you can "pop out" the lineage tree view into a separate window using the icon that looks like two boxes in the top left of the lineage tree view. You can also close the tree view with the x just above it, and open it again from the menu bar: Plugins -> Motile -> Lineage View (then re-run the below cell to add the data to the lineage view).

# %%
assign_tracklet_ids(gt_tracks)
ground_truth_run = MotileRun(
    run_name="ground_truth",
    tracks=gt_tracks,
)

widget.view_controller.update_napari_layers(ground_truth_run, time_attr="t", pos_attr=("x", "y"))

# %% [markdown]
# ## Build a candidate graph from the detections
#
# To set up our tracking problem, we will create a "candidate graph" - a DiGraph that contains all possible detections (graph nodes) and links (graph edges) between them.
#
# Then we use an optimization method called an integer linear program (ILP) to select the best nodes and edges from the candidate graph to generate our final tracks.
#
# To create our candidate graph, we will use the provided StarDist segmentations.
# Each node in the candidate graph represents one segmentation, and each edge represents a potential link between segmentations. This candidate graph will also contain features that will be used in the optimization task, such as position on nodes and, later, customized scores on edges.


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 2: Extract candidate nodes from the predicted segmentations</h3>
# First we need to turn each segmentation into a node in a `networkx.DiGraph`.
# Use <a href=https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>skimage.measure.regionprops</a> to extract properties from each segmentation, and create a candidate graph with nodes only.
#
#
# Here are the requirements for the output graph:
# <ol>
#     <li>Each detection (unique label id) in the segmentation becomes a node in the graph</li>
#     <li>The node id is the label of the detection</li>
#     <li>Each node has an integer "t" attribute, based on the index into the first dimension of the input segmentation array</li>
#     <li>Each node has float "x" and "y" attributes containing the "x" and "y" values from the centroid of the detection region</li>
#     <li>Each node has a "score" attribute containing the probability score output from StarDist. The probability map is at half resolution, so you will need to divide the centroid by 2 before indexing into the probability score.</li>
#     <li>The graph has no edges (yet!)</li>
# </ol>
# </div>

# %% tags=["task"]
def nodes_from_segmentation(segmentation: np.ndarray) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        seg_frame = segmentation[t]
        props = skimage.measure.regionprops(seg_frame)
        for regionprop in props:
            ### YOUR CODE HERE ###

    return cand_graph

cand_graph = nodes_from_segmentation(segmentation)


# %% tags=["solution"]
def nodes_from_segmentation(segmentation: np.ndarray) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        seg_frame = segmentation[t]
        props = skimage.measure.regionprops(seg_frame)
        for regionprop in props:
            node_id = regionprop.label
            x = float(regionprop.centroid[0])
            y = float(regionprop.centroid[1])
            attrs = {
                "t": t,
                "x": x,
                "y": y,
                "score": float(probabilities[t, int(x // 2), int(y // 2)]),
            }
            assert node_id not in cand_graph.nodes
            cand_graph.add_node(node_id, **attrs)
    return cand_graph

cand_graph = nodes_from_segmentation(segmentation)

# %%
# run this cell to test your implementation of the candidate graph
assert cand_graph.number_of_nodes() == 6123, f"Found {cand_graph.number_of_nodes()} nodes, expected 6123"
assert cand_graph.number_of_edges() == 0, f"Found {cand_graph.number_of_edges()} edges, expected 0"
for node, data in cand_graph.nodes(data=True):
    assert type(node) == int, f"Node id {node} has type {type(node)}, expected 'int'"
    assert "t" in data, f"'t' attribute missing for node {node}"
    assert type(data["t"]) == int, f"'t' attribute has type {type(data['t'])}, expected 'int'"
    assert "x" in data, f"'x' attribute missing for node {node}"
    assert type(data["x"]) == float, f"'x' attribute has type {type(data['x'])}, expected 'float'"
    assert "y" in data, f"'y' attribute missing for node {node}"
    assert type(data["y"]) == float, f"'y' attribute has type {type(data['y'])}, expected 'float'"
    assert "score" in data, f"'score' attribute missing for node {node}"
    assert type(data["score"]) == float, f"'score' attribute has type {type(data['score'])}, expected 'float'"
print("Your candidate graph passed all the tests!")

# %% [markdown]
# We can visualize our candidate points using the napari Points layer. You should see one point in the center of each segmentation when we display it using the below cell.

# %%
points_array = np.array([[data["t"], data["x"], data["y"]] for node, data in cand_graph.nodes(data=True)])
cand_points_layer = napari.layers.Points(data=points_array, name="cand_points")
viewer.add_layer(cand_points_layer)


# %% [markdown]
# ### Adding Candidate Edges
#
# After extracting the nodes, we need to add candidate edges. The `add_cand_edges` function below adds candidate edges to a nodes-only graph by connecting all nodes in adjacent frames that are closer than a given max_edge_distance.
#
# Note: At the bottom of the cell, we add edges to our candidate graph with max_edge_distance=50. This is the maximum number of pixels that a cell centroid will be able to move between frames. If you want longer edges to be possible, you can increase this distance, but solving may take longer.

# %%
def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["t"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict

def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"]] for node in node_ids]
    return scipy.spatial.KDTree(positions)

def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
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
    node_frame_dict = _compute_node_frame_dict(cand_graph)

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

add_cand_edges(cand_graph, max_edge_distance=50)

# %% [markdown]
# Visualizing the candidate edges in napari is, unfortunately, not yet possible. However, we can print out the number of candidate nodes and edges, and compare it to the ground truth nodes and edgesedges. We should see that we have a few more candidate nodes than ground truth (due to false positive detections) and many more candidate edges than ground truth - our next step will be to use optimization to pick a subset of the candidate nodes and edges to generate our solution tracks.

# %%
print(f"Our candidate graph has {cand_graph.number_of_nodes()} nodes and {cand_graph.number_of_edges()} edges")
print(f"Our ground truth track graph has {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()}")


# %% [markdown]
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have visualized our data in napari and set up a candidate graph with all possible detections and links that we could select with our optimization task. </h3>
#
# We will now together go through the `motile` <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example before you actually set up and run your own motile optimization. If you reach this checkpoint early, feel free to start reading through the quickstart and think of questions you want to ask!
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
# `motile` ([docs here](https://funkelab.github.io/motile/)), makes it easy to link with an ILP in python by implementing common linking constraints and costs.

# %% [markdown]
# ## Task 3 - Basic tracking with motile
# <div class="alert alert-block alert-info"><h3>Task 3: Set up a basic motile tracking pipeline</h3>
# <p>Use the motile <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example to set up a basic motile pipeline for our task.
#
# Here are some key similarities and differences between the quickstart and our task:
# <ul>
#     <li>We do not have scores on our edges. However, we can use the edge distance as a cost, so that longer edges are more costly than shorter edges. Instead of using the <code>EdgeSelection</code> cost, we can use the <a href=https://funkelab.github.io/motile/api.html#edgedistance><code>EdgeDistance</code></a> cost with <code>position_attribute="pos"</code>. You will want a positive weight, since higher distances should be more costly, unlike in the example when higher scores were good and so we inverted them with a negative weight.</li>
#     <li>Because distance is always positive, and you want a positive weight, you will want to include a negative constant on the <code>EdgeDistance</code> cost. If there are no negative selection costs, the ILP will always select nothing, because the cost of selecting nothing is zero.</li>
#     <li>We want to allow divisions. So, we should pass in 2 to our <code>MaxChildren</code> constraint. The <code>MaxParents</code> constraint should have 1, the same as the quickstart, because neither task allows merging.</li>
#     <li>You should include an <code>Appear</code> cost and a <code>NodeSelection</code> cost similar to the one in the quickstart.</li>
# </ul>
#
# Once you have set up the basic motile optimization task in the function below, you will probably need to adjust the weight and constant values on your costs until you get a solution that looks reasonable.
#
# </p>
# </div>
#

# %% tags=["task"]
def solve_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    ### YOUR CODE HERE ###
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph


# %% tags=["solution"]
def solve_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-1.0, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeDistance(weight=1, constant=-20, position_attribute=("x", "y"))
    )
    solver.add_cost(motile.costs.Appear(constant=2.0))
    solver.add_cost(motile.costs.Split(constant=1.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


# %% [markdown]
# Here is a utility function to gauge some statistics of a solution.

# %%
def print_graph_stats(graph, name):
    print(f"{name}\t\t{graph.number_of_nodes()} nodes\t{graph.number_of_edges()} edges\t{len(list(nx.weakly_connected_components(graph)))} tracks")


# %% [markdown]
# Here we actually run the optimization, and compare the found solution to the ground truth.
#
# <div class="alert alert-block alert-warning"><h3>Gurobi license error</h3>
# Please ignore the warning `Could not create Gurobi backend ...`.
#
#
# Our integer linear program (ILP) tries to use the proprietary solver Gurobi. You probably don't have a license, in which case the ILP will fall back to the open source solver SCIP.
#
# SCIP is slower than Gurobi - to deal with this, we add a 120 second timeout to the solve call, which should approximate the truly optimal solution. For larger problems, or cases where getting the most optimal solution is crucial, one could increase the timeout or get a Gurobi license (recommended).
# </div>

# %%
# run this cell to actually run the solving and get a solution
solution_graph = solve_basic_optimization(cand_graph)

# then print some statistics about the solution compared to the ground truth
print_graph_stats(solution_graph, "solution")
print_graph_stats(gt_tracks, "gt tracks")


# %% [markdown]
# If you haven't selected any nodes or edges in your solution, try adjusting your weight and/or constant values. Make sure you have some negative costs or selecting nothing will always be the best solution!

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 1: Interpret your results based on statistics</h3>
# <p>
# What do these printed statistics tell you about your solution? What else would you like to know?
# </p>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h3>Checkpoint 2</h3>
# We will discuss the exercise up to this point as a group shortly. If you reach this checkpoint early, you can go on to Checkpoint 3.
# </div>

# %% [markdown]
# ## Visualize the Result
# Rather than just looking at printed statistics about our solution, let's visualize it in `napari`.
#
# Before we can create our MotileRun, we need to create an output segmentation from our solution. Our output segmentation differs from our input segmentation in two main ways:
# 1. Not all candidate nodes will be selected in our solution graph. We need to filter the masks corresponding to the un-selected candidate detections out of the output segmentation.
# 2. Segmentations will be relabeled so that the same cell will be the same label (and thus color) over time. Cells will still change label/color at division.
#
# Note that bad tracking results at this point does not mean that you implemented anything wrong! We still need to customize our costs and constraints to the task before we can get good results. As long as your pipeline selects something, and you can kind of interepret why it is going wrong, that is all that is needed at this point.

# %%
def relabel_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results to get the output segmentation.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for relabeling. Nodes not in graph will be removed from seg.
        segmentation (np.ndarray): Original segmentation with dimensions (t,y,x)

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,y,x)
    """
    assign_tracklet_ids(solution_nx_graph)
    tracked_masks = np.zeros_like(segmentation)
    for node, data in solution_nx_graph.nodes(data=True):
        time_frame = solution_nx_graph.nodes[node]["t"]
        previous_seg_id = node
        track_id = solution_nx_graph.nodes[node]["tracklet_id"]
        previous_seg_mask = (
            segmentation[time_frame] == previous_seg_id
        )
        tracked_masks[time_frame][previous_seg_mask] = track_id
    return tracked_masks

solution_seg = relabel_segmentation(solution_graph, segmentation)

# %%
basic_run = MotileRun(
    run_name="basic_solution",
    tracks=solution_graph,
    output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
)

widget.view_controller.update_napari_layers(basic_run, time_attr="t", pos_attr=("x", "y"))

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 2: Interpret your results based on visualization</h3>
# <p>
# How is your solution based on looking at the visualization? When is it doing well? When is it doing poorly?
# </p>
# </div>
#

# %% [markdown]
# ## Evaluation Metrics
#
# We were able to understand via visualizing the predicted tracks on the images that the basic solution is far from perfect for this problem.
#
# Additionally, we would also like to quantify this. We will use the package [`traccuracy`](https://traccuracy.readthedocs.io/en/latest/) to calculate some [standard metrics for cell tracking](http://celltrackingchallenge.net/evaluation-methodology/). For this exercise, we'll take a look at the following metrics:
#
# - **TRA**: TRA is a metric established by the [Cell Tracking Challenge](http://celltrackingchallenge.net). It compares your solution graph to the ground truth graph and measures how many changes to edges and nodes would need to be made in order to make the graphs identical. TRA ranges between 0 and 1 with 1 indicating a perfect match between the solution and the ground truth. While TRAf is convenient to use in that it gives us a single number, it doesn't tell us what type of mistakes are being made in our solution.
# - **Node Errors**: We can look at the number of false positive and false negative nodes in our solution which tells us how how many cells are being incorrectly included or excluded from the solution.
# - **Edge Errors**: Similarly, the number of false positive and false negative edges in our graph helps us assess what types of mistakes our solution is making when linking cells between frames.
# - **Division Errors**: Finally, as biologists we are often very interested in the division events that occur and want to ensure that they are being accurately identified. We can look at the number of true positive, false positive and false negative divisions to assess how our solution is capturing these important events.


# %% [markdown]
# The metrics we want to compute require a ground truth segmentation. Since we do not have a ground truth segmentation, we can make one by drawing a circle around each ground truth detection. While not perfect, it will be good enough to match ground truth to predicted detections in order to compute metrics.

# %%
from skimage.draw import disk
def make_gt_detections(data_shape, gt_tracks, radius):
    segmentation = np.zeros(data_shape, dtype="uint32")
    frame_shape = data_shape[1:]
    # make frame with one cell in center with label 1
    for node, data in gt_tracks.nodes(data=True):
        pos = (data["x"], data["y"])
        time = data["t"]
        gt_tracks.nodes[node]["label"] = node
        rr, cc = disk(center=pos, radius=radius, shape=frame_shape)
        segmentation[time][rr, cc] = node
    return segmentation

gt_dets = make_gt_detections(data_root["raw"].shape, gt_tracks, 10)

# %%
import pandas as pd

def get_metrics(gt_graph, labels, run, results_df):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        run (MotileRun): Instance of Motilerun
        results_df (pd.DataFrame): Dataframe containing any prior results

    Returns:
        results (pd.DataFrame): Dataframe of evaluation results
    """
    gt_graph = traccuracy.TrackingGraph(
        graph=gt_graph,
        frame_key="t",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=labels,
    )

    pred_graph = traccuracy.TrackingGraph(
        graph=run.tracks,
        frame_key="t",
        label_key="tracklet_id",
        location_keys=("x", "y"),
        segmentation=np.squeeze(run.output_segmentation),
    )

    results = traccuracy.run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=IOUMatcher(iou_threshold=0.3, one_to_one=True),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )
    columns = ["fp_nodes", "fn_nodes", "fp_edges", "fn_edges", "TRA", "True Positive Divisions", "False Positive Divisions", "False Negative Divisions"]
    results_filtered = {}
    results_filtered.update(results[0]["results"])
    results_filtered.update(results[1]["results"]["Frame Buffer 0"])
    results_filtered["name"] = run.run_name
    current_result = pd.DataFrame(results_filtered, index=[0])[["name"] + columns]

    if results_df is None:
        results_df = current_result
    else:
        results_df = pd.concat([results_df, current_result])

    return results_df


# %%
results_df = None
results_df = get_metrics(gt_tracks, gt_dets, basic_run, results_df)
results_df


# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 3: Interpret your results based on metrics</h3>
# <p>
# What additional information, if any, do the metrics give you compared to the statistics and the visualization?
# </p>
# </div>
#

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 3</h2>
# If you reach this checkpoint with extra time, think about what kinds of improvements you could make to the costs and constraints to fix the issues that you are seeing. You can try tuning your weights and constants, or adding or removing motile Costs and Constraints, and seeing how that changes the output. We have added a convenience function in the box below where you can copy your solution from above, adapt it, and run the whole pipeline including visualizaiton and metrics computation.
#
# Do not get frustrated if you cannot get good results yet! Try to think about why and what custom costs we might add.
# </div>

# %% tags=["task"]
def adapt_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    ### YOUR CODE HERE ###
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph

def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = adapt_basic_optimization(cand_graph)
    solution_seg = relabel_segmentation(solution_graph, segmentation)
    run = MotileRun(
        run_name=run_name,
        tracks=solution_graph,
        output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
    )
    widget.view_controller.update_napari_layers(run, time_attr="t", pos_attr=("x", "y"))
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run below, so you can tell them apart in the results table
results_df = run_pipeline(cand_graph, "basic_solution_2", results_df)
results_df
   

# %% tags=["solution"]
def adapt_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-5.0, constant=2.5, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeDistance(weight=1, constant=-20, position_attribute=("x", "y"))
    )
    solver.add_cost(motile.costs.Appear(constant=20.0))
    solver.add_cost(motile.costs.Split(constant=15.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph

def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = adapt_basic_optimization(cand_graph)
    solution_seg = relabel_segmentation(solution_graph, segmentation)
    run = MotileRun(
        run_name=run_name,
        tracks=solution_graph,
        output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
    )
    widget.view_controller.update_napari_layers(run, time_attr="t", pos_attr=("x", "y"))
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

results_df = run_pipeline(cand_graph, "basic_solution_2", results_df)
results_df
   

# %% [markdown]
# ## Customizing the Tracking Task
#
# There 3 main ways to encode prior knowledge about your task into the motile tracking pipeline.
# 1. Add an attribute to the candidate graph and incorporate it with an existing cost
# 2. Change the structure of the candidate graph
# 3. Add a new type of cost or constraint
#
# The first way is the most common, and is quite flexible, so we will focus on an example of this type of customization.

# %% [markdown]
# ## Task 4 - Incorporating Known Direction of Motion
#
# So far, we have been using motile's EdgeDistance as an edge selection cost, which penalizes longer edges by computing the Euclidean distance between the endpoints. However, in our dataset we see a trend of upward motion in the cells, and the false detections at the top are not moving. If we penalize movement based on what we expect, rather than Euclidean distance, we can select more correct cells and penalize the non-moving artefacts at the same time.
#
#

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4a: Add a drift distance attribute</h3>
# <p> For this task, we need to determine the "expected" amount of motion, then add an attribute to our candidate edges that represents distance from the expected motion direction.</p>
# </div>

# %% tags=["task"]
drift = ... ### YOUR CODE HERE ###

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        ### YOUR CODE HERE ###
        # get the location of the endpoints of the edge
        # then compute the distance between the expected movement and the actual movement
        # and save it in the "drift_dist" attribute (below)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)

# %% tags=["solution"]
drift = np.array([-10, 0])

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        source, target = edge
        source_data = cand_graph.nodes[source]
        source_pos = np.array([source_data["x"], source_data["y"]])
        target_data = cand_graph.nodes[target]
        target_pos = np.array([target_data["x"], target_data["y"]])
        expected_target_pos = source_pos + drift
        drift_dist = np.linalg.norm(expected_target_pos - target_pos)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4b: Add a drift distance attribute</h3>
# <p> Now, we set up yet another solving pipeline. This time, we will replace our EdgeDistance
# cost with an EdgeSelection cost using our new "drift_dist" attribute. The weight should be positive, since a higher distance from the expected drift should cost more, similar to our prior EdgeDistance cost. Also similarly, we need a negative constant to make sure that the overall cost of selecting tracks is negative.</p>
# </div>

# %% tags=["task"]
def solve_drift_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)

    ### YOUR CODE HERE ###

    solver.solve(timeout=120)

    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_drift_optimization(cand_graph)
    solution_seg = relabel_segmentation(solution_graph, segmentation)
    run = MotileRun(
        run_name=run_name,
        tracks=solution_graph,
        output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
    )
    widget.view_controller.update_napari_layers(run, time_attr="t", pos_attr=("x", "y"))
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "drift_dist", results_df)
results_df


# %% tags=["solution"]
def solve_drift_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-100, constant=75, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeSelection(weight=1.0, constant=-30, attribute="drift_dist")
    )
    solver.add_cost(motile.costs.Appear(constant=40.0))
    solver.add_cost(motile.costs.Split(constant=45.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_drift_optimization(cand_graph)
    solution_seg = relabel_segmentation(solution_graph, segmentation)
    run = MotileRun(
        run_name=run_name,
        tracks=solution_graph,
        output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
    )
    widget.view_controller.update_napari_layers(run, time_attr="t", pos_attr=("x", "y"))
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "node_const_75", results_df)
results_df


# %% [markdown]
# Feel free to tinker with the weights and constants manually to try and improve the results.
# You should be able to get something decent now, but this dataset is quite difficult! There are still many custom costs that could be added to improve the results - we will discuss some ideas together shortly.

# %% [markdown]
# <div class="alert alert-block alert-success"><h3>Checkpoint 4</h3>
# That is the end of the main exercise! If you have extra time, feel free to go onto the below bonus exercise to see how to learn the weights of your costs instead of setting them manually.
# </div>

# %% [markdown]
# ## Bonus: Learning the Weights

# %% [markdown]
# Motile also provides the option to learn the best weights and constants using a [Structured Support Vector Machine](https://en.wikipedia.org/wiki/Structured_support_vector_machine). There is a tutorial on the motile documentation [here](https://funkelab.github.io/motile/learning.html), but we will also walk you through an example below.
#
# We need some ground truth annotations on our candidate graph in order to learn the best weights. The next cell contains a function that matches our ground truth graph to our candidate graph using the predicted segmentations. The function checks for each ground truth node if it is inside one of our predicted segmentations. If it is, that candidate node is marked with attribute "gt" = True. Any unmatched candidate nodes have "gt" = False. We also annotate the edges in a similar fashion - if both endpoints of a GT edge are inside predicted segmentations, the corresponding candidate edge will have "gt" = True, while all other edges going out of that candidate node have "gt" = False.

# %%
def get_cand_id(gt_node, gt_track, cand_segmentation):
    data = gt_track.nodes[gt_node]
    return cand_segmentation[data["t"], int(data["x"])][int(data["y"])]

def add_gt_annotations(gt_tracks, cand_graph, segmentation):
    for gt_node in gt_tracks.nodes():
        cand_id = get_cand_id(gt_node, gt_tracks, segmentation)
        if cand_id != 0:
            if cand_id in cand_graph:
                cand_graph.nodes[cand_id]["gt"] = True
                gt_succs = gt_tracks.successors(gt_node)
                gt_succ_matches = [get_cand_id(gt_succ, gt_tracks, segmentation) for gt_succ in gt_succs]
                cand_succs = cand_graph.successors(cand_id)
                for succ in cand_succs:
                    if succ in gt_succ_matches:
                        cand_graph.edges[(cand_id, succ)]["gt"] = True
                    else:
                        cand_graph.edges[(cand_id, succ)]["gt"] = False
    for node in cand_graph.nodes():
       if "gt" not in cand_graph.nodes[node]:
           cand_graph.nodes[node]["gt"] = False


# %% [markdown]
# The SSVM does not need dense ground truth - providing only some annotations frequently is sufficient to learn good weights, and is efficient for both computation time and annotation time. Below, we create a validation graph that spans the first three time frames, and annotate it with our ground truth.

# %%
validation_times = [0, 3]
validation_nodes = [node for node, data in cand_graph.nodes(data=True)
                        if (data["t"] >= validation_times[0] and data["t"] < validation_times[1])]
print(len(validation_nodes))
validation_graph = cand_graph.subgraph(validation_nodes).copy()
add_gt_annotations(gt_tracks, validation_graph, segmentation)


# %% [markdown]
# Here we print the number of nodes and edges that have been annotated with True and False ground truth. It is important to provide negative/False annotations, as well as positive/True annotations, or the SSVM will try and select weights to pick everything.

# %%
gt_pos_nodes = [node_id for node_id, data in validation_graph.nodes(data=True) if "gt" in data and data["gt"] is True]
gt_neg_nodes = [node_id for node_id, data in validation_graph.nodes(data=True) if "gt" in data and data["gt"] is False]
gt_pos_edges = [(source, target) for source, target, data in validation_graph.edges(data=True) if "gt" in data and data["gt"] is True]
gt_neg_edges = [(source, target) for source, target, data in validation_graph.edges(data=True) if "gt" in data and data["gt"] is False]

print(f"{len(gt_pos_nodes) + len(gt_neg_nodes)} annotated: {len(gt_pos_nodes)} True, {len(gt_neg_nodes)} False")
print(f"{len(gt_pos_edges) + len(gt_neg_edges)} annotated: {len(gt_pos_edges)} True, {len(gt_neg_edges)} False")


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Bonus task: Add your best solver parameters</h3>
# <p>Now, similar to before, we make the solver by adding costs and constraints. You can copy your best set of costs and constraints from before. It does not matter what weights and constants you choose. However, this time we just return the solver, rather than actually solving.</p>
# </div>

# %% tags=["task"]
def get_ssvm_solver(cand_graph):

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)

    ### YOUR CODE HERE ###
    return solver


# %%
def get_ssvm_solver(cand_graph):

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-1.0, attribute='score')
    )
    solver.add_cost(
        motile.costs.EdgeSelection(weight=1.0, constant=-30, attribute="drift_dist")
    )
    solver.add_cost(motile.costs.Split(constant=20))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))
    return solver


# %% [markdown]
# To fit the best weights, the solver will solve the ILP many times and slowly converge to the best set of weights in a structured manner. Running the cell below may take some time - we recommend getting a Gurobi license if you want to use this technique in your research, as it speeds up solving quite a bit.
#
# At the end, it will print the optimal weights, and you can compare them to the weights you found by trial and error.

# %%
ssvm_solver = get_ssvm_solver(validation_graph)
ssvm_solver.fit_weights(gt_attribute="gt", regularizer_weight=100, max_iterations=50)
optimal_weights = ssvm_solver.weights
optimal_weights


# %% [markdown]
# After we have our optimal weights, we need to solve with them on the full candidate graph.

# %%
def get_ssvm_solution(cand_graph, solver_weights):
    solver = get_ssvm_solver(cand_graph)
    solver.weights = solver_weights
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph

solution_graph = get_ssvm_solution(cand_graph, optimal_weights)


# %% [markdown]
# Finally, we can visualize and compute metrics on the solution found using the weights discovered by the SSVM.

# %%
solution_seg = relabel_segmentation(solution_graph, segmentation)

# %%
ssvm_run = MotileRun(
    run_name="ssvm_solution",
    tracks=solution_graph,
    output_segmentation=np.expand_dims(solution_seg, axis=1)  # need to add a dummy dimension to fit API
)

widget.view_controller.update_napari_layers(ssvm_run, time_attr="t", pos_attr=("x", "y"))

# %%

results_df = get_metrics(gt_tracks, gt_dets, ssvm_run, results_df)
results_df

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Bonus Question: Interpret SSVM results</h3>
# <p>
# How do the results compare between the SSVM-discovered weights and your hand-crafted weights? What are the advantages and disadvantages of each approach in terms of (human or computer) time needed?
# </p>
# </div>
#
