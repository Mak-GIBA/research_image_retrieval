"""
Table 1 implementations for image retrieval methods.

This package contains implementations of all methods listed in Table 1 of the paper
"ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval".

Available models:
- GeM pooling (R50/R101)
- DELG (R50/R101)
- Token-based (R50/R101)
- HOW-VLAD (R50/R101)
- HOW-ASMK (R50/R101)
- SENet-G2+ (50/101)
- SoSNet (R50/R101)
- SpoC (R50/R101)
"""

from .gem_pooling import GeMWrapper, get_gem_model, create_gem_optimizer
from .delg import DELGWrapper, get_delg_model, create_delg_optimizer
from .token_based import TokenWrapper, get_token_model, create_token_optimizer
from .how_vlad import HOWVLADWrapper, HOWASMKWrapper, get_how_vlad_model, get_how_asmk_model, create_how_optimizer
from .senet_g2 import SENetG2Wrapper, get_senet_g2_model, create_senet_g2_optimizer
from .sosnet import SoSNetWrapper, get_sosnet_model, create_sosnet_optimizer
from .spoc import SpoCWrapper, get_spoc_model, create_spoc_optimizer

from .wrappers import (
    get_model, 
    get_optimizer, 
    list_available_models, 
    get_model_info,
    get_model_from_table1_name,
    get_optimizer_from_table1_name,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    TABLE1_TO_MODEL_MAPPING
)

__all__ = [
    # Individual model wrappers
    'GeMWrapper',
    'DELGWrapper', 
    'TokenWrapper',
    'HOWVLADWrapper',
    'HOWASMKWrapper',
    'SENetG2Wrapper',
    'SoSNetWrapper',
    'SpoCWrapper',
    
    # Model factory functions
    'get_gem_model',
    'get_delg_model',
    'get_token_model',
    'get_how_vlad_model',
    'get_how_asmk_model',
    'get_senet_g2_model',
    'get_sosnet_model',
    'get_spoc_model',
    
    # Optimizer factory functions
    'create_gem_optimizer',
    'create_delg_optimizer',
    'create_token_optimizer',
    'create_how_optimizer',
    'create_senet_g2_optimizer',
    'create_sosnet_optimizer',
    'create_spoc_optimizer',
    
    # Unified interface
    'get_model',
    'get_optimizer',
    'list_available_models',
    'get_model_info',
    'get_model_from_table1_name',
    'get_optimizer_from_table1_name',
    
    # Registries
    'MODEL_REGISTRY',
    'OPTIMIZER_REGISTRY',
    'TABLE1_TO_MODEL_MAPPING',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Table 1 Implementation Team'
__description__ = 'Implementations of image retrieval methods from Table 1'

