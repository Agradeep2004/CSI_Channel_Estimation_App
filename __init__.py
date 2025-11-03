"""
CSI Channel Estimation App
==========================

A modular Python package for wireless channel estimation using
machine learning and deep learning models (CNN, LSTM, CVNN, Transformer).

This __init__.py initializes global paths, logging, and key imports
for the Streamlit app, making the modules importable as a unified package.

Author: Agradeep Sarkar
Institute: SRM Institute of Science and Technology
Year: 2025
"""

import os
import sys
import logging
from pathlib import Path

# -------------------------------------------------------------------------
# Project Metadata
# -------------------------------------------------------------------------
__appname__ = "CSI_Channel_Estimation_App"
__version__ = "1.0.0"
__author__ = "Agradeep Sarkar"
__license__ = "MIT"
__description__ = "Machine Learning-based Wireless Channel Estimation Dashboard"

# -------------------------------------------------------------------------
# Directory Setup
# -------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed_data_fixed"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if missing
for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------------
# Add project root to PYTHONPATH (for module imports)
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# -------------------------------------------------------------------------
# Convenient Imports for Submodules
# -------------------------------------------------------------------------
try:
    from . import data_gen
    from . import models
except ImportError:
    logging.warning("Data and Model modules not found — ensure structure is correct.")

# -------------------------------------------------------------------------
# Utility Function for Global Access
# -------------------------------------------------------------------------
def get_app_info():
    """Return app metadata as a dictionary."""
    return {
        "name": __appname__,
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "data_path": str(DATA_DIR),
        "results_path": str(RESULTS_DIR),
        "models_path": str(MODELS_DIR),
    }

logging.info("CSI Channel Estimation App successfully initialized.")
