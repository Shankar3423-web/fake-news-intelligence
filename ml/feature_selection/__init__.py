"""
Phase 5: Feature Selection Module.
This module is responsible for reducing the feature space of dense and sparse (TF-IDF) features 
by running a pipeline of variance thresholds, correlation filters, mutual information, 
chi-square, random forest importance, and recursive feature elimination.
"""

from ml.feature_selection.selection_config import SelectionConfig
from ml.feature_selection.selection_logger import setup_logger
from ml.feature_selection.selection_utils import BaseFeatureSelector, load_data, compute_file_sha256, get_memory_usage
from ml.feature_selection.variance_selector import VarianceSelector
from ml.feature_selection.correlation_selector import CorrelationSelector
from ml.feature_selection.mutual_information_selector import MutualInformationSelector
from ml.feature_selection.chi_square_selector import ChiSquareSelector
from ml.feature_selection.random_forest_selector import RandomForestSelector
from ml.feature_selection.rfe_selector import RFESelector
from ml.feature_selection.selector_merger import SelectorMerger
from ml.feature_selection.selection_validator import SelectionValidator
from ml.feature_selection.selection_statistics import SelectionStatistics
from ml.feature_selection.selection_profiler import SelectionProfiler
from ml.feature_selection.feature_selection_pipeline import run_feature_selection_pipeline

__all__ = [
    "SelectionConfig",
    "setup_logger",
    "BaseFeatureSelector",
    "load_data",
    "compute_file_sha256",
    "get_memory_usage",
    "VarianceSelector",
    "CorrelationSelector",
    "MutualInformationSelector",
    "ChiSquareSelector",
    "RandomForestSelector",
    "RFESelector",
    "SelectorMerger",
    "SelectionValidator",
    "SelectionStatistics",
    "SelectionProfiler",
    "run_feature_selection_pipeline"
]
