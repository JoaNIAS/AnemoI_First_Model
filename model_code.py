import datetime
import numpy as np
import zarr
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import knn_graph
import pandas as pd
from anemoi.datasets import open_dataset
import os

# Sets environment variables for ECMWF API authentication.
os.environ["ECMWF_API_KEY"] = ""
os.environ["ECMWF_API_EMAIL"] = ""
os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"

# Creates a new dataset using a recipe1.yaml file 
!anemoi-datasets create recipe1.yaml dataset_new_onemore.zarr --overwrite

# Creates a graph structure from the specified recipe and saves it as a .pt file.
!anemoi-graphs create graph_recipe1.yaml graph1.pt

# Inspects the graph file and generates output plots in the specified directory.
!anemoi-graphs inspect graph1.pt output_plots

# Trains a model using the specified configuration file with full error logging enabled.
!HYDRA_FULL_ERROR=1 anemoi-training train --config-name=config_default.yaml
