"""
Unified wrappers and model factory for all Table 1 methods.
This module provides a unified interface to create and use all implemented models.
"""

import torch
import torch.nn as nn
from .gem_pooling import get_gem_model, create_gem_optimizer
from .delg import get_delg_model, create_delg_optimizer
from .token_based import get_token_model, create_token_optimizer
from .how_vlad import get_how_vlad_model, get_how_asmk_model, create_how_optimizer
from .senet_g2 import get_senet_g2_model, create_senet_g2_optimizer
from .sosnet import get_sosnet_model, create_sosnet_optimizer
from .spoc import get_spoc_model, create_spoc_optimizer


# Model registry
MODEL_REGISTRY = {
    # GeM-based models
    'gem_r50': lambda num_classes, **kwargs: get_gem_model(num_classes, backbone='resnet50', **kwargs),
    'gem_r101': lambda num_classes, **kwargs: get_gem_model(num_classes, backbone='resnet101', **kwargs),
    
    # DELG-based models
    'delg_r50': lambda num_classes, **kwargs: get_delg_model(num_classes, backbone='resnet50', **kwargs),
    'delg_r101': lambda num_classes, **kwargs: get_delg_model(num_classes, backbone='resnet101', **kwargs),
    
    # Token-based models
    'token_r50': lambda num_classes, **kwargs: get_token_model(num_classes, backbone='resnet50', **kwargs),
    'token_r101': lambda num_classes, **kwargs: get_token_model(num_classes, backbone='resnet101', **kwargs),
    
    # HOW-VLAD models
    'how_vlad_r50': lambda num_classes, **kwargs: get_how_vlad_model(num_classes, backbone='resnet50', **kwargs),
    'how_vlad_r101': lambda num_classes, **kwargs: get_how_vlad_model(num_classes, backbone='resnet101', **kwargs),
    
    # HOW-ASMK models
    'how_asmk_r50': lambda num_classes, **kwargs: get_how_asmk_model(num_classes, backbone='resnet50', **kwargs),
    'how_asmk_r101': lambda num_classes, **kwargs: get_how_asmk_model(num_classes, backbone='resnet101', **kwargs),
    
    # SENet-G2+ models
    'senet_g2_50': lambda num_classes, **kwargs: get_senet_g2_model(num_classes, backbone='senet50', **kwargs),
    'senet_g2_101': lambda num_classes, **kwargs: get_senet_g2_model(num_classes, backbone='senet101', **kwargs),
    
    # SoSNet models
    'sosnet_r50': lambda num_classes, **kwargs: get_sosnet_model(num_classes, backbone='resnet50', **kwargs),
    'sosnet_r101': lambda num_classes, **kwargs: get_sosnet_model(num_classes, backbone='resnet101', **kwargs),
    
    # SpoC models
    'spoc_r50': lambda num_classes, **kwargs: get_spoc_model(num_classes, backbone='resnet50', **kwargs),
    'spoc_r101': lambda num_classes, **kwargs: get_spoc_model(num_classes, backbone='resnet101', **kwargs),
}


# Optimizer registry
OPTIMIZER_REGISTRY = {
    'gem_r50': create_gem_optimizer,
    'gem_r101': create_gem_optimizer,
    'delg_r50': create_delg_optimizer,
    'delg_r101': create_delg_optimizer,
    'token_r50': create_token_optimizer,
    'token_r101': create_token_optimizer,
    'how_vlad_r50': create_how_optimizer,
    'how_vlad_r101': create_how_optimizer,
    'how_asmk_r50': create_how_optimizer,
    'how_asmk_r101': create_how_optimizer,
    'senet_g2_50': create_senet_g2_optimizer,
    'senet_g2_101': create_senet_g2_optimizer,
    'sosnet_r50': create_sosnet_optimizer,
    'sosnet_r101': create_sosnet_optimizer,
    'spoc_r50': create_spoc_optimizer,
    'spoc_r101': create_spoc_optimizer,
}


def get_model(model_name, num_classes, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name (str): Name of the model to create
        num_classes (int): Number of classes for classification
        **kwargs: Additional arguments for model creation
    
    Returns:
        torch.nn.Module: The created model
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name](num_classes, **kwargs)


def get_optimizer(model_name, args, model):
    """
    Factory function to create optimizers by model name.
    
    Args:
        model_name (str): Name of the model
        args: Arguments containing optimizer parameters
        model (torch.nn.Module): The model to optimize
    
    Returns:
        torch.optim.Optimizer: The created optimizer
    """
    if model_name not in OPTIMIZER_REGISTRY:
        # Default to SGD if no specific optimizer is defined
        return torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=getattr(args, 'weight_decay', 1e-4),
            nesterov=True
        )
    
    return OPTIMIZER_REGISTRY[model_name](args, model)


def list_available_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name):
    """
    Get information about a specific model.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Information about the model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Extract information from model name
    parts = model_name.split('_')
    method = parts[0]
    
    if len(parts) > 1:
        if parts[1].startswith('r'):
            backbone = f"resnet{parts[1][1:]}"
        elif parts[1] == 'g2':
            backbone = f"senet{parts[2]}"
        else:
            backbone = parts[1]
    else:
        backbone = "resnet50"  # default
    
    info = {
        'method': method,
        'backbone': backbone,
        'full_name': model_name,
        'description': _get_method_description(method)
    }
    
    return info


def _get_method_description(method):
    """Get description for each method."""
    descriptions = {
        'gem': 'Generalized Mean Pooling for image retrieval',
        'delg': 'Deep Local and Global features for image search',
        'token': 'Token-based transformer approach for image retrieval',
        'how': 'Hyperdimensional Object Whitening with VLAD/ASMK pooling',
        'senet': 'SENet backbone with G2+ pooling',
        'sosnet': 'Second-order Similarity Network',
        'spoc': 'Spatial Pyramid of Contexts'
    }
    return descriptions.get(method, f'{method} method for image retrieval')


# Mapping from Table 1 names to our model names
TABLE1_TO_MODEL_MAPPING = {
    # Group (c): Global single pass models
    'R101-GeM': 'gem_r101',
    'R101-DELG': 'delg_r101',
    'R50-Token': 'token_r50',
    'R101-Token': 'token_r101',
    'R50-SENet-G2+': 'senet_g2_50',
    'R101-SENet-G2+': 'senet_g2_101',
    'R50-SoSNet': 'sosnet_r50',
    'R101-SoSNet': 'sosnet_r101',
    'R101-SpoC': 'spoc_r101',
    
    # Group (b): Local features aggregation
    'R101-HOW-VLAD': 'how_vlad_r101',
    'R101-HOW-ASMK': 'how_asmk_r101',
}


def get_model_from_table1_name(table1_name, num_classes, **kwargs):
    """
    Create model using Table 1 naming convention.
    
    Args:
        table1_name (str): Model name as it appears in Table 1
        num_classes (int): Number of classes
        **kwargs: Additional arguments
    
    Returns:
        torch.nn.Module: The created model
    """
    if table1_name not in TABLE1_TO_MODEL_MAPPING:
        available_names = list(TABLE1_TO_MODEL_MAPPING.keys())
        raise ValueError(f"Unknown Table 1 model: {table1_name}. Available: {available_names}")
    
    model_name = TABLE1_TO_MODEL_MAPPING[table1_name]
    return get_model(model_name, num_classes, **kwargs)


def get_optimizer_from_table1_name(table1_name, args, model):
    """
    Create optimizer using Table 1 naming convention.
    
    Args:
        table1_name (str): Model name as it appears in Table 1
        args: Arguments containing optimizer parameters
        model (torch.nn.Module): The model to optimize
    
    Returns:
        torch.optim.Optimizer: The created optimizer
    """
    if table1_name not in TABLE1_TO_MODEL_MAPPING:
        # Default optimizer
        return torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=getattr(args, 'weight_decay', 1e-4),
            nesterov=True
        )
    
    model_name = TABLE1_TO_MODEL_MAPPING[table1_name]
    return get_optimizer(model_name, args, model)

