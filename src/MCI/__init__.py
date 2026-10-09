from os.path import abspath
import datetime
import numpy as np
x = datetime.datetime.now()

VERSION = '1.0'
print('MCI library version {} @ {} date: {}'.format(VERSION, abspath(__file__), x))
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, inplace=True)
from .write_dataset import write_dataset
from .MCI import MCI
from .MCI_assoc import MCI_assoc
from .get_1d_array_unique_objects_in_order import get_1d_array_unique_objects_in_order
from .cluster_2d_array_slices import cluster_2d_array_slices
from .match_target_vs_features import match_target_vs_features
from .association_matrix2 import association_matrix2
from .compute_ranks_and_copula_dens_multigrid_v2 import compute_ranks_and_copula_dens_multigrid_v2
from .compute_copula_entropies import compute_copula_entropies
from .compute_empirical_p_value import compute_empirical_p_value
from .check_nd_array_for_bad import check_nd_array_for_bad
from ._ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays import _ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays
