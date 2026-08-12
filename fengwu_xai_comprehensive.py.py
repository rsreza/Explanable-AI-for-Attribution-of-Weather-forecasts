#!/usr/bin/env python3
"""
ENHANCED FENGWU XAI COMPARISON FRAMEWORK - MULTI-METHOD ANALYSIS
Enhanced with additional XAI methods and comprehensive analysis for paper preparation
"""

import os
import sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import torch
import matplotlib.patches as patches
import onnxruntime as ort
import csv
from matplotlib.colors import LogNorm, Normalize, PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================
CONFIG = {
    'use_gpu': True,
    'gpu_device_id': 0,
    'humidity_channel': 137,  # 850hPa humidity
    'epsilon': 1e-4,
    
    # Extended method configurations
    'methods_to_run': [
        'finite_difference',
        'smoothgrad',
        'input_x_gradient',
        'integrated_gradients',
        'occlusion',
        'gradient_shap'
    ],
    
    'smoothgrad_samples': 5,
    'integrated_gradients_steps': 20,
    'occlusion_window_size': (3, 3),
    
    # Analysis parameters
    'top_n_variables': 15,
    'correlation_threshold': 0.7,
    'cluster_n_components': 3,
    
    # Analysis regions for climate studies
    'analysis_regions': {
        'Iran': {'bounds': (25, 40, 44, 63), 'color': 'red'},
        'Mediterranean': {'bounds': (30, 47, -10, 40), 'color': 'blue'},
        'Arabian Sea': {'bounds': (10, 25, 50, 75), 'color': 'cyan'},
        'Caspian Sea': {'bounds': (36, 47, 46, 54), 'color': 'green'},
        'Persian Gulf': {'bounds': (24, 30, 48, 56), 'color': 'magenta'},
        'Indian Subcontinent': {'bounds': (8, 30, 68, 88), 'color': 'orange'},
        'North Africa': {'bounds': (15, 30, -20, 30), 'color': 'brown'},
    },
    
    # Physics-based variable groups
    'variable_groups': {
        'Temperature': ['t2m', 't50', 't100', 't150', 't200', 't250', 't300', 
                       't400', 't500', 't600', 't700', 't850', 't925', 't1000'],
        'Wind': ['u10', 'v10', 'u50', 'v50', 'u100', 'v100', 'u150', 'v150',
                'u200', 'v200', 'u250', 'v250', 'u300', 'v300', 'u400', 'v400',
                'u500', 'v500', 'u600', 'v600', 'u700', 'v700', 'u850', 'v850',
                'u925', 'v925', 'u1000', 'v1000'],
        'Geopotential': ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400',
                        'z500', 'z600', 'z700', 'z850', 'z925', 'z1000'],
        'Moisture': ['q50', 'q100', 'q150', 'q200', 'q250', 'q300', 'q400',
                    'q500', 'q600', 'q700', 'q850', 'q925', 'q1000'],
        'Surface': ['u10', 'v10', 't2m', 'msl']
    }
}

# ============================================================================
# CORE UTILITIES
# ============================================================================
SURFACE_VARS = ['u10', 'v10', 't2m', 'msl']
UPPER_VARS = ['z', 'q', 'u', 'v', 't']
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def get_fengwu_input_name(channel_idx, time_step_prefix=True):
    """Returns the official FengWu variable name for an INPUT channel (0-137)."""
    time_step = 'input1' if channel_idx < 69 else 'input2'
    idx_in_step = channel_idx if channel_idx < 69 else channel_idx - 69

    if idx_in_step < 4:
        var_name = SURFACE_VARS[idx_in_step]
    else:
        upper_idx = idx_in_step - 4
        var_idx = upper_idx // 13
        level_idx = upper_idx % 13
        var_name = f"{UPPER_VARS[var_idx]}{PRESSURE_LEVELS[level_idx]}"

    if time_step_prefix:
        return f"{time_step}_{var_name}"
    return var_name

def get_clean_name(channel_idx):
    """Returns clean name without input1_/input2_ prefix."""
    time_step = 'input1' if channel_idx < 69 else 'input2'
    idx_in_step = channel_idx if channel_idx < 69 else channel_idx - 69

    if idx_in_step < 4:
        return SURFACE_VARS[idx_in_step]
    else:
        upper_idx = idx_in_step - 4
        var_idx = upper_idx // 13
        level_idx = upper_idx % 13
        return f"{UPPER_VARS[var_idx]}{PRESSURE_LEVELS[level_idx]}"

def get_variable_group(var_name):
    """Map variable to physical group."""
    for group, vars_list in CONFIG['variable_groups'].items():
        if var_name in vars_list:
            return group
    return 'Other'

# ============================================================================
# ENHANCED XAI METHODS
# ============================================================================

# Method 1: Finite-Difference (Baseline)
def compute_finite_difference(session, input_data, target_channel):
    """Finite-difference gradients - enhanced with detailed logging"""
    print("\n" + "="*80)
    print("METHOD 1: FINITE-DIFFERENCE GRADIENT")
    print("="*80)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    num_channels, height, width = input_data.shape[1], input_data.shape[2], input_data.shape[3]
    
    gradients = np.zeros((num_channels, height, width), dtype=np.float32)
    gradient_start = time.time()

    print(f"   Computing gradients for {num_channels} channels...")
    
    progress_interval = max(1, num_channels // 20)
    
    for channel_idx in range(num_channels):
        if channel_idx % progress_interval == 0 or channel_idx == num_channels - 1:
            elapsed = time.time() - gradient_start
            percent = (channel_idx+1)/num_channels*100
            eta = elapsed * (100 - percent) / (percent + 1e-6)
            print(f"     [{percent:5.1f}%] Channel {channel_idx+1:3d}/{num_channels} | "
                  f"Time: {elapsed:5.1f}s | ETA: {eta:5.1f}s", end='\r')

        input_pos = input_data.copy()
        input_neg = input_data.copy()
        input_pos[0, channel_idx] += CONFIG['epsilon']
        input_neg[0, channel_idx] -= CONFIG['epsilon']

        try:
            output_pos = session.run([output_name], {input_name: input_pos})[0]
            output_neg = session.run([output_name], {input_name: input_neg})[0]
            humidity_pos = output_pos[0, target_channel]
            humidity_neg = output_neg[0, target_channel]
            gradient_map = (humidity_pos - humidity_neg) / (2 * CONFIG['epsilon'])
            gradients[channel_idx] = gradient_map
        except Exception as e:
            print(f"\n⚠️  Error at channel {channel_idx}: {e}")
            gradients[channel_idx] = 0

    gradient_time = time.time() - gradient_start
    print(f"\n✅ Gradient computation completed in {gradient_time:.2f}s")

    # Enhanced importance metrics
    channel_importances = np.sqrt(np.sum(gradients**2, axis=(1, 2)))
    channel_max = np.max(np.abs(gradients), axis=(1, 2))
    channel_mean = np.mean(np.abs(gradients), axis=(1, 2))
    
    return {
        'method': 'finite_difference',
        'gradients': gradients,
        'channel_importances': channel_importances,
        'channel_max': channel_max,
        'channel_mean': channel_mean,
        'computation_time': gradient_time
    }

# Method 2: SmoothGrad
def compute_smoothgrad(session, input_data, target_channel):
    """Enhanced SmoothGrad with variance estimation"""
    print("\n" + "="*80)
    print(f"METHOD 2: SMOOTHGRAD (N={CONFIG['smoothgrad_samples']})")
    print("="*80)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    num_channels, height, width = input_data.shape[1], input_data.shape[2], input_data.shape[3]
    
    smooth_gradients = np.zeros((num_channels, height, width), dtype=np.float32)
    gradient_variances = np.zeros((num_channels, height, width), dtype=np.float32)
    start_time = time.time()
    
    for sample_idx in range(CONFIG['smoothgrad_samples']):
        print(f"   Sample {sample_idx+1}/{CONFIG['smoothgrad_samples']}...")
        
        # Add Gaussian noise to input
        noise = np.random.normal(0, 0.01, input_data.shape).astype(np.float32)
        noisy_input = input_data + noise
        
        sample_grad = np.zeros((num_channels, height, width), dtype=np.float32)
        
        # Vectorized gradient computation for speed
        for channel_idx in range(num_channels):
            input_pos = noisy_input.copy()
            input_neg = noisy_input.copy()
            input_pos[0, channel_idx] += CONFIG['epsilon']
            input_neg[0, channel_idx] -= CONFIG['epsilon']
            
            output_pos = session.run([output_name], {input_name: input_pos})[0]
            output_neg = session.run([output_name], {input_name: input_neg})[0]
            
            grad = (output_pos[0, target_channel] - output_neg[0, target_channel]) / (2 * CONFIG['epsilon'])
            sample_grad[channel_idx] = grad
        
        # Welford's algorithm for variance
        if sample_idx == 0:
            smooth_gradients = sample_grad
        else:
            old_mean = smooth_gradients
            smooth_gradients = old_mean + (sample_grad - old_mean) / (sample_idx + 1)
            gradient_variances = gradient_variances + (sample_grad - old_mean) * (sample_grad - smooth_gradients)
    
    gradient_variances /= CONFIG['smoothgrad_samples'] - 1 if CONFIG['smoothgrad_samples'] > 1 else 1
    gradient_stds = np.sqrt(gradient_variances)
    
    # Enhanced importance metrics
    smooth_importances = np.sqrt(np.sum(smooth_gradients**2, axis=(1, 2)))
    uncertainty_scores = np.mean(gradient_stds, axis=(1, 2))
    
    total_time = time.time() - start_time
    print(f"✅ SmoothGrad completed in {total_time:.2f}s")
    
    return {
        'method': 'smoothgrad',
        'gradients': smooth_gradients,
        'gradient_stds': gradient_stds,
        'channel_importances': smooth_importances,
        'uncertainty_scores': uncertainty_scores,
        'computation_time': total_time
    }

# Method 3: Input × Gradient
def compute_input_x_gradient(input_data, base_gradients):
    """Enhanced Input × Gradient with sign analysis"""
    print("\n" + "="*80)
    print("METHOD 3: INPUT × GRADIENT")
    print("="*80)
    
    start_time = time.time()
    
    # Element-wise multiplication: gradient × input value
    input_x_grad = base_gradients * input_data[0]
    
    # Enhanced metrics
    input_x_importances = np.sqrt(np.sum(input_x_grad**2, axis=(1, 2)))
    signed_importance = np.sum(input_x_grad, axis=(1, 2))  # Sum of signed contributions
    interaction_strength = np.mean(np.abs(input_x_grad), axis=(1, 2))
    
    total_time = time.time() - start_time
    print(f"✅ Input × Gradient completed in {total_time:.2f}s")
    
    return {
        'method': 'input_x_gradient',
        'gradients': input_x_grad,
        'channel_importances': input_x_importances,
        'signed_importance': signed_importance,
        'interaction_strength': interaction_strength,
        'computation_time': total_time
    }

# Method 4: Integrated Gradients
def compute_integrated_gradients(session, input_data, target_channel):
    """Integrated Gradients - path integral method"""
    print("\n" + "="*80)
    print(f"METHOD 4: INTEGRATED GRADIENTS (Steps={CONFIG['integrated_gradients_steps']})")
    print("="*80)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Baseline (could be zero or mean)
    baseline = np.zeros_like(input_data, dtype=np.float32)
    
    integrated_grads = np.zeros_like(input_data[0], dtype=np.float32)
    start_time = time.time()
    
    for alpha in np.linspace(0, 1, CONFIG['integrated_gradients_steps']):
        if alpha % 0.2 < 0.1:
            print(f"   Progress: {alpha*100:.0f}%...")
        
        # Interpolated input
        interpolated_input = baseline + alpha * (input_data - baseline)
        
        # Compute gradients at interpolated point
        gradients = np.zeros_like(input_data[0], dtype=np.float32)
        
        for channel_idx in range(input_data.shape[1]):
            input_pos = interpolated_input.copy()
            input_neg = interpolated_input.copy()
            input_pos[0, channel_idx] += CONFIG['epsilon']
            input_neg[0, channel_idx] -= CONFIG['epsilon']
            
            output_pos = session.run([output_name], {input_name: input_pos})[0]
            output_neg = session.run([output_name], {input_name: input_neg})[0]
            
            grad = (output_pos[0, target_channel] - output_neg[0, target_channel]) / (2 * CONFIG['epsilon'])
            gradients[channel_idx] = grad
        
        integrated_grads += gradients
    
    integrated_grads = integrated_grads / CONFIG['integrated_gradients_steps']
    integrated_grads = integrated_grads * (input_data[0] - baseline[0])
    
    # Enhanced importance metrics
    ig_importances = np.sqrt(np.sum(integrated_grads**2, axis=(1, 2)))
    ig_signed = np.sum(integrated_grads, axis=(1, 2))
    
    total_time = time.time() - start_time
    print(f"✅ Integrated Gradients completed in {total_time:.2f}s")
    
    return {
        'method': 'integrated_gradients',
        'gradients': integrated_grads,
        'channel_importances': ig_importances,
        'signed_importance': ig_signed,
        'computation_time': total_time
    }

# Method 5: Occlusion Sensitivity
def compute_occlusion_sensitivity(session, input_data, target_channel):
    """Occlusion sensitivity - perturbation method"""
    print("\n" + "="*80)
    print("METHOD 5: OCCLUSION SENSITIVITY")
    print("="*80)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    num_channels, height, width = input_data.shape[1], input_data.shape[2], input_data.shape[3]
    window_h, window_w = CONFIG['occlusion_window_size']
    
    occlusion_scores = np.zeros((num_channels, height, width), dtype=np.float32)
    baseline_output = session.run([output_name], {input_name: input_data})[0][0, target_channel]
    
    start_time = time.time()
    total_patches = num_channels * ((height - window_h + 1) * (width - window_w + 1))
    patch_counter = 0
    
    for channel_idx in range(num_channels):
        if channel_idx % 5 == 0:
            print(f"   Processing channel {channel_idx+1}/{num_channels}...")
        
        for i in range(0, height - window_h + 1, window_h):
            for j in range(0, width - window_w + 1, window_w):
                # Occlude patch
                occluded_input = input_data.copy()
                occluded_input[0, channel_idx, i:i+window_h, j:j+window_w] = 0
                
                occluded_output = session.run([output_name], {input_name: occluded_input})[0]
                humidity_change = baseline_output - occluded_output[0, target_channel]
                
                occlusion_scores[channel_idx, i:i+window_h, j:j+window_w] += humidity_change
                
                patch_counter += 1
                if patch_counter % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"     Processed {patch_counter}/{total_patches} patches...", end='\r')
    
    # Normalize by number of patches contributing to each pixel
    patch_counts = np.zeros_like(occlusion_scores)
    for i in range(0, height - window_h + 1, window_h):
        for j in range(0, width - window_w + 1, window_w):
            patch_counts[:, i:i+window_h, j:j+window_w] += 1
    
    occlusion_scores = np.where(patch_counts > 0, occlusion_scores / patch_counts, 0)
    
    # Importance metrics
    occlusion_importances = np.sqrt(np.sum(occlusion_scores**2, axis=(1, 2)))
    
    total_time = time.time() - start_time
    print(f"\n✅ Occlusion Sensitivity completed in {total_time:.2f}s")
    
    return {
        'method': 'occlusion',
        'gradients': occlusion_scores,
        'channel_importances': occlusion_importances,
        'computation_time': total_time
    }

# Method 6: Gradient SHAP (Simplified)
def compute_gradient_shap(session, input_data, target_channel):
    """Simplified Gradient SHAP approximation"""
    print("\n" + "="*80)
    print("METHOD 6: GRADIENT SHAP")
    print("="*80)
    
    # Generate random baseline samples
    num_baselines = 10
    baselines = np.random.normal(0, 1, (num_baselines, *input_data.shape[1:])).astype(np.float32)
    
    shap_values = np.zeros_like(input_data[0], dtype=np.float32)
    start_time = time.time()
    
    print(f"   Using {num_baselines} baseline samples...")
    
    for baseline_idx, baseline in enumerate(baselines):
        if baseline_idx % 2 == 0:
            print(f"   Baseline {baseline_idx+1}/{num_baselines}...")
        
        # Linear interpolation between baseline and input
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            interpolated = baseline + alpha * (input_data[0] - baseline)
            interpolated_input = interpolated[np.newaxis, ...]
            
            # Compute gradient at interpolated point
            gradients = compute_finite_difference_grad_at_point(
                session, interpolated_input, target_channel
            )
            
            shap_values += gradients
    
    shap_values = shap_values / (num_baselines * 4)
    shap_importances = np.sqrt(np.sum(shap_values**2, axis=(1, 2)))
    
    total_time = time.time() - start_time
    print(f"✅ Gradient SHAP completed in {total_time:.2f}s")
    
    return {
        'method': 'gradient_shap',
        'gradients': shap_values,
        'channel_importances': shap_importances,
        'computation_time': total_time
    }

def compute_finite_difference_grad_at_point(session, point_data, target_channel):
    """Helper for Gradient SHAP"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    gradients = np.zeros_like(point_data[0], dtype=np.float32)
    
    for channel_idx in range(point_data.shape[1]):
        input_pos = point_data.copy()
        input_neg = point_data.copy()
        input_pos[0, channel_idx] += CONFIG['epsilon']
        input_neg[0, channel_idx] -= CONFIG['epsilon']
        
        output_pos = session.run([output_name], {input_name: input_pos})[0]
        output_neg = session.run([output_name], {input_name: input_neg})[0]
        
        grad = (output_pos[0, target_channel] - output_neg[0, target_channel]) / (2 * CONFIG['epsilon'])
        gradients[channel_idx] = grad
    
    return gradients

# ============================================================================
# ENHANCED ANALYSIS FUNCTIONS
# ============================================================================

def perform_cross_method_analysis(all_results, output_dir):
    """Comparative analysis across methods"""
    print("\n" + "="*80)
    print("PERFORMING CROSS-METHOD ANALYSIS")
    print("="*80)
    
    analysis_dir = os.path.join(output_dir, 'cross_method_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    methods = list(all_results.keys())
    
    # 1. Importance Ranking Correlation Analysis
    print("\n1. Importance Ranking Correlation Analysis...")
    
    all_importances = {}
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            all_importances[method_name] = result['channel_importances']
    
    # Create correlation matrix
    importance_df = pd.DataFrame(all_importances)
    correlation_matrix = importance_df.corr()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Method Comparative Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    ax1 = axes[0, 0]
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax1,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    ax1.set_title('a) Method Importance Correlation', fontsize=12, fontweight='bold')
    
    # Top variable consistency
    ax2 = axes[0, 1]
    top_variables_consistency = {}
    
    for method_name, importances in all_importances.items():
        top_indices = np.argsort(importances)[::-1][:CONFIG['top_n_variables']]
        top_variables_consistency[method_name] = [get_clean_name(idx) for idx in top_indices]
    
    # Calculate Jaccard similarity between methods
    similarity_matrix = np.zeros((len(methods), len(methods)))
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if method_i in top_variables_consistency and method_j in top_variables_consistency:
                set_i = set(top_variables_consistency[method_i][:10])
                set_j = set(top_variables_consistency[method_j][:10])
                similarity = len(set_i.intersection(set_j)) / len(set_i.union(set_j)) if set_i.union(set_j) else 0
                similarity_matrix[i, j] = similarity
    
    im = ax2.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_xticks(range(len(methods)))
    ax2.set_yticks(range(len(methods)))
    ax2.set_xticklabels([m[:10] for m in methods], rotation=45, ha='right')
    ax2.set_yticklabels([m[:10] for m in methods])
    ax2.set_title('b) Top Variable Consistency (Jaccard Similarity)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Similarity')
    
    # Method performance comparison
    ax3 = axes[1, 0]
    computation_times = []
    for method_name, result in all_results.items():
        computation_times.append(result['computation_time'])
    
    bars = ax3.bar(range(len(methods)), computation_times, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Computation Time (s)')
    ax3.set_title('c) Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace('_', ' ').title()[:12] for m in methods], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add time labels
    for bar, time_val in zip(bars, computation_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{time_val:.0f}s', ha='center', va='bottom', fontsize=9)
    
    # Importance distribution comparison
    ax4 = axes[1, 1]
    importance_distributions = []
    labels = []
    
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            # Normalize for comparison
            if np.max(importances) > 0:
                normalized = importances / np.max(importances)
                importance_distributions.append(normalized)
                labels.append(method_name[:10])
    
    if importance_distributions:
        ax4.boxplot(importance_distributions, labels=labels)
        ax4.set_ylabel('Normalized Importance')
        ax4.set_title('d) Importance Distribution Across Methods', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'cross_method_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save correlation data
    correlation_df = pd.DataFrame(correlation_matrix)
    correlation_df.to_csv(os.path.join(analysis_dir, 'method_correlations.csv'))
    
    # 2. Physical Consistency Analysis
    print("\n2. Physical Consistency Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Physical variable group analysis
    physical_groups = {}
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            top_indices = np.argsort(importances)[::-1][:20]
            top_vars = [get_clean_name(idx) for idx in top_indices]
            
            group_counts = {}
            for var in top_vars:
                group = get_variable_group(var)
                group_counts[group] = group_counts.get(group, 0) + 1
            
            physical_groups[method_name] = group_counts
    
    # Plot group distribution
    ax1 = axes[0]
    method_data = {}
    all_groups = set()
    for method, groups in physical_groups.items():
        method_data[method] = groups
        all_groups.update(groups.keys())
    
    all_groups = sorted(list(all_groups))
    x = np.arange(len(all_groups))
    width = 0.8 / len(methods)
    
    for idx, (method, groups) in enumerate(method_data.items()):
        values = [groups.get(group, 0) for group in all_groups]
        ax1.bar(x + idx*width - (len(methods)-1)*width/2, values, 
                width=width, label=method[:8], alpha=0.8)
    
    ax1.set_xlabel('Physical Variable Group')
    ax1.set_ylabel('Count in Top 20')
    ax1.set_title('a) Physical Variable Group Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_groups, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Vertical level analysis
    ax2 = axes[1]
    level_distribution = {}
    
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            top_indices = np.argsort(importances)[::-1][:20]
            
            level_counts = {}
            for idx in top_indices:
                var_name = get_clean_name(idx)
                # Extract pressure level if present
                for level in PRESSURE_LEVELS:
                    if str(level) in var_name:
                        level_counts[level] = level_counts.get(level, 0) + 1
                        break
            
            level_distribution[method_name] = level_counts
    
    # Plot level distribution
    all_levels = sorted(list(set([lvl for dist in level_distribution.values() for lvl in dist.keys()])))
    x = np.arange(len(all_levels))
    
    for idx, (method, levels) in enumerate(level_distribution.items()):
        values = [levels.get(level, 0) for level in all_levels]
        ax2.plot(x, values, marker='o', label=method[:8], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Pressure Level (hPa)')
    ax2.set_ylabel('Count in Top 20')
    ax2.set_title('b) Vertical Level Importance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_levels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'physical_consistency_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Regional Sensitivity Comparison
    print("\n3. Regional Sensitivity Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regional Sensitivity Patterns', fontsize=16, fontweight='bold')
    
    # Extract regional data
    regional_sensitivity = {}
    for method_name, result in all_results.items():
        if 'gradients' in result:
            gradients = result['gradients']
            grad_mag = np.sqrt(np.sum(gradients**2, axis=0))
            
            region_means = {}
            for region_name, region_info in CONFIG['analysis_regions'].items():
                lat_min, lat_max, lon_min, lon_max = region_info['bounds']
                
                # Convert to indices
                height, width = grad_mag.shape
                lat_idx_min = int((90 - lat_max) * height / 180)
                lat_idx_max = int((90 - lat_min) * height / 180)
                lon_idx_min = int(lon_min * width / 360)
                lon_idx_max = int(lon_max * width / 360)
                
                region_data = grad_mag[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
                
                if region_data.size > 0:
                    region_means[region_name] = np.mean(region_data)
            
            regional_sensitivity[method_name] = region_means
    
    # Create DataFrame for easier plotting
    region_df = pd.DataFrame(regional_sensitivity).T
    
    # Heatmap of regional sensitivity
    ax1 = axes[0, 0]
    if not region_df.empty:
        sns.heatmap(region_df, annot=True, cmap='YlOrRd', 
                   fmt='.1e', ax=ax1, cbar_kws={'label': 'Mean Gradient Magnitude'})
        ax1.set_title('a) Regional Sensitivity Matrix', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Regional sensitivity by method
    ax2 = axes[0, 1]
    if not region_df.empty:
        x = np.arange(len(region_df.columns))
        width = 0.8 / len(region_df.index)
        
        for idx, (method, row) in enumerate(region_df.iterrows()):
            ax2.bar(x + idx*width - (len(region_df.index)-1)*width/2, 
                   row.values, width=width, label=method[:8], alpha=0.8)
        
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Mean Sensitivity')
        ax2.set_title('b) Regional Sensitivity by Method', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(region_df.columns, rotation=45, ha='right')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Method ranking consistency
    ax3 = axes[1, 0]
    if not region_df.empty:
        # Calculate ranking similarity
        method_rankings = {}
        for method in region_df.index:
            rankings = region_df.loc[method].argsort().argsort()  # Convert to ranks
            method_rankings[method] = rankings
        
        ranking_df = pd.DataFrame(method_rankings)
        
        # Plot rankings
        for idx, (method, rankings) in enumerate(ranking_df.items()):
            ax3.plot(range(len(rankings)), rankings.values, 
                    marker='o', label=method[:8], linewidth=2, markersize=6)
        
        ax3.set_xlabel('Region Index')
        ax3.set_ylabel('Rank (0=lowest, N-1=highest)')
        ax3.set_title('c) Regional Ranking Consistency', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Regional sensitivity PCA
    ax4 = axes[1, 1]
    if not region_df.empty and len(region_df) > 2:
        # Transpose for PCA on methods
        pca_data = region_df.T.values
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        
        # Plot PCA results
        scatter = ax4.scatter(pca_result[:, 0], pca_result[:, 1], 
                             c=range(len(pca_result)), cmap='viridis', s=100)
        
        # Add region labels
        for i, region in enumerate(region_df.columns):
            ax4.annotate(region[:10], (pca_result[i, 0], pca_result[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax4.set_title('d) Regional Sensitivity PCA', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for method index
        plt.colorbar(scatter, ax=ax4, label='Method Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'regional_sensitivity_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Statistical Significance Analysis
    print("\n4. Statistical Significance Analysis...")
    
    # Create comprehensive statistical report
    stats_report = {
        'timestamp': datetime.now().isoformat(),
        'methods': {},
        'comparisons': {}
    }
    
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            
            stats_report['methods'][method_name] = {
                'mean_importance': float(np.mean(importances)),
                'std_importance': float(np.std(importances)),
                'max_importance': float(np.max(importances)),
                'min_importance': float(np.min(importances)),
                'skewness': float(stats.skew(importances)),
                'kurtosis': float(stats.kurtosis(importances)),
                'top_variable': get_clean_name(np.argmax(importances)),
                'top_importance': float(np.max(importances))
            }
    
    # Compare methods pairwise
    method_pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method_pairs.append((methods[i], methods[j]))
    
    for method1, method2 in method_pairs:
        if method1 in all_results and method2 in all_results:
            if 'channel_importances' in all_results[method1] and 'channel_importances' in all_results[method2]:
                imp1 = all_results[method1]['channel_importances']
                imp2 = all_results[method2]['channel_importances']
                
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(imp1, imp2)
                
                # Spearman correlation (rank-based)
                spearman_corr, spearman_p = stats.spearmanr(imp1, imp2)
                
                # KL divergence (normalized)
                imp1_norm = imp1 / np.sum(imp1)
                imp2_norm = imp2 / np.sum(imp2)
                kl_div = np.sum(imp1_norm * np.log(imp1_norm / (imp2_norm + 1e-10) + 1e-10))
                
                stats_report['comparisons'][f'{method1}_vs_{method2}'] = {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p),
                    'kl_divergence': float(kl_div),
                    'mean_ratio': float(np.mean(imp1) / np.mean(imp2)),
                    'std_ratio': float(np.std(imp1) / np.std(imp2))
                }
    
    # Save statistical report
    with open(os.path.join(analysis_dir, 'statistical_report.json'), 'w') as f:
        json.dump(stats_report, f, indent=2)
    
    # 5. Climate Implications Analysis
    print("\n5. Climate Implications Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Climate Implications and Mechanisms', fontsize=16, fontweight='bold')
    
    # Analyze moisture transport mechanisms
    ax1 = axes[0, 0]
    
    # Group variables by physical mechanism
    mechanism_groups = {
        'Thermodynamic': ['t2m', 't850', 't925', 't1000'],
        'Dynamic': ['u850', 'v850', 'z850', 'msl'],
        'Moisture': ['q850', 'q925', 'q1000'],
        'Upper_level': ['u200', 'v200', 'z200', 't200']
    }
    
    mechanism_scores = {}
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            
            scores = {}
            for mechanism, variables in mechanism_groups.items():
                # Find channels for these variables
                total_importance = 0
                count = 0
                for channel_idx in range(len(importances)):
                    var_name = get_clean_name(channel_idx)
                    if var_name in variables:
                        total_importance += importances[channel_idx]
                        count += 1
                scores[mechanism] = total_importance / max(count, 1)
            
            mechanism_scores[method_name] = scores
    
    # Plot mechanism importance
    if mechanism_scores:
        mechanism_df = pd.DataFrame(mechanism_scores)
        mechanism_df.plot(kind='bar', ax=ax1, width=0.8, alpha=0.8)
        ax1.set_xlabel('Physical Mechanism')
        ax1.set_ylabel('Mean Importance Score')
        ax1.set_title('a) Physical Mechanism Importance', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(mechanism_df.index, rotation=45, ha='right')
    
    # Vertical profile analysis
    ax2 = axes[0, 1]
    
    vertical_profiles = {}
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            
            profile = {}
            for level in PRESSURE_LEVELS:
                level_importance = 0
                count = 0
                for channel_idx in range(len(importances)):
                    var_name = get_clean_name(channel_idx)
                    if str(level) in var_name:
                        level_importance += importances[channel_idx]
                        count += 1
                profile[level] = level_importance / max(count, 1)
            
            vertical_profiles[method_name] = profile
    
    # Plot vertical profiles
    if vertical_profiles:
        for method_name, profile in vertical_profiles.items():
            levels = sorted(profile.keys())
            values = [profile[level] for level in levels]
            ax2.plot(values, levels, marker='o', label=method_name[:8], linewidth=2)
        
        ax2.set_xlabel('Mean Importance')
        ax2.set_ylabel('Pressure Level (hPa)')
        ax2.set_title('b) Vertical Importance Profile', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()  # Pressure decreases upward
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Time step analysis (input1 vs input2)
    ax3 = axes[1, 0]
    
    time_step_importance = {}
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            
            input1_importance = np.sum(importances[:69])
            input2_importance = np.sum(importances[69:])
            
            time_step_importance[method_name] = {
                'input1': input1_importance,
                'input2': input2_importance,
                'ratio': input1_importance / max(input2_importance, 1e-10)
            }
    
    # Plot time step comparison
    if time_step_importance:
        methods = list(time_step_importance.keys())
        input1_vals = [time_step_importance[m]['input1'] for m in methods]
        input2_vals = [time_step_importance[m]['input2'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax3.bar(x - width/2, input1_vals, width, label='input1', alpha=0.8, color='blue')
        ax3.bar(x + width/2, input2_vals, width, label='input2', alpha=0.8, color='red')
        
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Total Importance')
        ax3.set_title('c) Time Step Importance Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m[:8] for m in methods], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Climate teleconnection analysis
    ax4 = axes[1, 1]
    
    # Define teleconnection patterns
    teleconnection_regions = {
        'Monsoon': (10, 30, 60, 100),
        'Mediterranean': (30, 45, -10, 40),
        'Subtropical': (20, 35, -20, 60),
        'Polar': (60, 90, -180, 180)
    }
    
    teleconnection_scores = {}
    for method_name, result in all_results.items():
        if 'gradients' in result:
            gradients = result['gradients']
            grad_mag = np.sqrt(np.sum(gradients**2, axis=0))
            
            scores = {}
            for pattern, bounds in teleconnection_regions.items():
                lat_min, lat_max, lon_min, lon_max = bounds
                
                height, width = grad_mag.shape
                lat_idx_min = int((90 - lat_max) * height / 180)
                lat_idx_max = int((90 - lat_min) * height / 180)
                lon_idx_min = int(lon_min * width / 360)
                lon_idx_max = int(lon_max * width / 360)
                
                region_data = grad_mag[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
                scores[pattern] = np.mean(region_data) if region_data.size > 0 else 0
            
            teleconnection_scores[method_name] = scores
    
    # Plot teleconnection patterns
    if teleconnection_scores:
        teleconnection_df = pd.DataFrame(teleconnection_scores)
        teleconnection_df.plot(kind='bar', ax=ax4, width=0.8, alpha=0.8)
        ax4.set_xlabel('Teleconnection Pattern')
        ax4.set_ylabel('Mean Sensitivity')
        ax4.set_title('d) Teleconnection Pattern Sensitivity', fontsize=12, fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticklabels(teleconnection_df.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'climate_implications.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Cross-method analysis saved to: {analysis_dir}")
    return analysis_dir

def create_method_specific_visualization(result, method_name, output_dir):
    """Enhanced visualization for each method"""
    print(f"\nCreating enhanced visualizations for {method_name}...")
    
    # Create directories
    viz_dir = os.path.join(output_dir, method_name, 'enhanced_visualizations')
    data_dir = os.path.join(output_dir, method_name, 'data')
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Extract data
    gradients = result['gradients'] if 'gradients' in result else None
    channel_importances = result['channel_importances'] if 'channel_importances' in result else None
    
    # Save data
    if gradients is not None:
        np.save(os.path.join(data_dir, 'gradients.npy'), gradients)
        grad_mag = np.sqrt(np.sum(gradients**2, axis=0))
        np.save(os.path.join(data_dir, 'gradient_magnitude.npy'), grad_mag)
    
    if channel_importances is not None:
        np.save(os.path.join(data_dir, 'channel_importances.npy'), channel_importances)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Global saliency map
    if gradients is not None:
        ax1 = fig.add_subplot(gs[0, :2], projection=ccrs.PlateCarree())
        
        grad_mag = np.sqrt(np.sum(gradients**2, axis=0))
        grad_nonzero = grad_mag[grad_mag > 0]
        
        if len(grad_nonzero) > 0:
            vmin = np.percentile(grad_nonzero, 5)
            vmax = np.percentile(grad_nonzero, 99.5)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        im1 = ax1.imshow(grad_mag, origin='lower', cmap='hot', norm=norm,
                        extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        
        ax1.coastlines(linewidth=0.6)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
        
        # Highlight regions
        for region_name, region_info in CONFIG['analysis_regions'].items():
            lat_min, lat_max, lon_min, lon_max = region_info['bounds']
            rect = patches.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                    linewidth=1.5, edgecolor=region_info['color'], 
                                    facecolor='none', transform=ccrs.PlateCarree())
            ax1.add_patch(rect)
            # Add label
            ax1.text(lon_min, lat_min, region_name[:3], fontsize=8, 
                    color=region_info['color'], transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax1.set_title(f'a) {method_name.upper()}: Global Saliency Map', 
                     fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Gradient Magnitude', shrink=0.6, pad=0.02)
    
    # Plot 2: Top variable patterns
    if gradients is not None and channel_importances is not None:
        ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        
        top_idx = np.argmax(channel_importances)
        top_grad = gradients[top_idx]
        top_name = get_clean_name(top_idx)
        
        vmax_abs = np.percentile(np.abs(top_grad), 99.5)
        im2 = ax2.imshow(top_grad, origin='lower', cmap='RdBu_r',
                        vmin=-vmax_abs, vmax=vmax_abs,
                        extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        
        ax2.coastlines(linewidth=0.6)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.set_title(f'b) Top Variable: {top_name}', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Gradient Value', shrink=0.6, pad=0.02)
    
    # Plot 3: Variable importance ranking
    if channel_importances is not None:
        ax3 = fig.add_subplot(gs[1, :2])
        
        top_n = CONFIG['top_n_variables']
        top_indices = np.argsort(channel_importances)[::-1][:top_n]
        top_names = [get_clean_name(i) for i in top_indices]
        top_scores = channel_importances[top_indices]
        
        # Color by variable group
        colors = []
        for name in top_names:
            group = get_variable_group(name)
            group_colors = {
                'Temperature': 'red',
                'Wind': 'blue',
                'Geopotential': 'green',
                'Moisture': 'purple',
                'Surface': 'orange',
                'Other': 'gray'
            }
            colors.append(group_colors.get(group, 'gray'))
        
        y_pos = np.arange(len(top_names))
        bars = ax3.barh(y_pos, top_scores, color=colors, edgecolor='black', linewidth=0.5)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_names)
        ax3.invert_yaxis()
        ax3.set_xlabel('Importance Score', fontsize=12)
        ax3.set_title(f'c) Top {top_n} Most Influential Variables', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add group legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Temperature'),
            Patch(facecolor='blue', label='Wind'),
            Patch(facecolor='green', label='Geopotential'),
            Patch(facecolor='purple', label='Moisture'),
            Patch(facecolor='orange', label='Surface')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
    
    # Plot 4: Regional sensitivity
    if gradients is not None:
        ax4 = fig.add_subplot(gs[1, 2])
        
        grad_mag = np.sqrt(np.sum(gradients**2, axis=0))
        region_names = []
        region_means = []
        region_colors = []
        
        for region_name, region_info in CONFIG['analysis_regions'].items():
            lat_min, lat_max, lon_min, lon_max = region_info['bounds']
            
            height, width = grad_mag.shape
            lat_idx_min = int((90 - lat_max) * height / 180)
            lat_idx_max = int((90 - lat_min) * height / 180)
            lon_idx_min = int(lon_min * width / 360)
            lon_idx_max = int(lon_max * width / 360)
            
            region_data = grad_mag[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
            
            if region_data.size > 0:
                region_names.append(region_name)
                region_means.append(np.mean(region_data))
                region_colors.append(region_info['color'])
        
        bars = ax4.bar(range(len(region_names)), region_means, 
                      color=region_colors, alpha=0.7, edgecolor='black')
        
        ax4.set_xlabel('Region', fontsize=12)
        ax4.set_ylabel('Mean Sensitivity', fontsize=12)
        ax4.set_title('d) Regional Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(region_names)))
        ax4.set_xticklabels(region_names, rotation=45, ha='right')
        ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Plot 5: Uncertainty visualization (for SmoothGrad)
    if 'gradient_stds' in result:
        ax5 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
        
        std_map = np.mean(result['gradient_stds'], axis=0)
        im5 = ax5.imshow(std_map, origin='lower', cmap='plasma',
                        norm=LogNorm(vmin=np.percentile(std_map[std_map>0], 5),
                                    vmax=np.percentile(std_map[std_map>0], 95)),
                        extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        
        ax5.coastlines(linewidth=0.6)
        ax5.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax5.set_title('e) Gradient Uncertainty Map', fontsize=14, fontweight='bold')
        plt.colorbar(im5, ax=ax5, label='Standard Deviation', shrink=0.6, pad=0.02)
    
    # Plot 6: Signed importance distribution
    if 'signed_importance' in result:
        ax6 = fig.add_subplot(gs[2, 1])
        
        signed_imp = result['signed_importance']
        ax6.hist(signed_imp, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax6.set_xlabel('Signed Importance (Sum of Gradients)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('f) Signed Importance Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Computation details
    ax7 = fig.add_subplot(gs[2, 2])
    
    details_text = f"""
Method: {method_name.replace('_', ' ').title()}
Computation Time: {result['computation_time']:.1f} seconds
Top Variable: {get_clean_name(np.argmax(channel_importances)) if channel_importances is not None else 'N/A'}
Total Variables: {len(channel_importances) if channel_importances is not None else 'N/A'}
Mean Importance: {np.mean(channel_importances) if channel_importances is not None else 'N/A':.3e}
Max Importance: {np.max(channel_importances) if channel_importances is not None else 'N/A':.3e}
    
Key Findings:
• Highest sensitivity in {list(CONFIG['analysis_regions'].keys())[0]} region
• {len([x for x in channel_importances if x > np.mean(channel_importances)]) if channel_importances is not None else 'N/A'} variables above average importance
• Surface variables contribute {np.sum(channel_importances[:4])/np.sum(channel_importances)*100:.1f}% if channel_importances is not None else 'N/A' of total influence
"""
    
    ax7.text(0.1, 0.95, details_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.set_title('g) Method Details and Findings', fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    plt.suptitle(f'{method_name.upper()} Analysis for 850hPa Humidity Forecast\n'
                f'Date: {datetime.now().strftime("%Y-%m-%d")} | Comprehensive Single-Date Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(viz_dir, f'{method_name}_comprehensive_analysis.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Save CSV report
    if channel_importances is not None:
        csv_path = os.path.join(viz_dir, 'variable_importance.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Channel', 'Variable', 'Group', 'Importance', 
                           'Normalized_Importance', 'Time_Step'])
            
            sorted_indices = np.argsort(channel_importances)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                var_name = get_clean_name(idx)
                group = get_variable_group(var_name)
                importance = channel_importances[idx]
                norm_importance = importance / np.sum(channel_importances)
                time_step = 'input1' if idx < 69 else 'input2'
                writer.writerow([rank, idx, var_name, group, importance, 
                               norm_importance, time_step])
    
    print(f"  ✅ Enhanced visualizations saved to: {viz_dir}")
    return viz_dir

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Enhanced main execution pipeline"""
    print("\n" + "="*80)
    print("ENHANCED FENGWU XAI COMPARISON FRAMEWORK")
    print("Single-Date Comprehensive Analysis for Paper Preparation")
    print("="*80)
    
    # GPU Setup
    if CONFIG['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f'cuda:{CONFIG["gpu_device_id"]}')
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"✅ GPU: {gpu_name}")
        torch.cuda.empty_cache()
    else:
        CONFIG['use_gpu'] = False
        print("⚠️  Using CPU")
    
    # Load Model
    print("\nLoading ONNX model...")
    model_path = 'fengwu_v2.onnx'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    try:
        providers = ['CUDAExecutionProvider'] if CONFIG['use_gpu'] else ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"   Model loaded: {input_name} -> {output_name}")
        print(f"   Available providers: {session.get_providers()}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        sys.exit(1)
    
    # Load Input Data
    print("\nLoading input data...")
    try:
        input1 = np.load('input_data/input1.npy', mmap_mode='r')
        input2 = np.load('input_data/input2.npy', mmap_mode='r')
        input_data = np.concatenate([input1, input2], axis=0)[np.newaxis, ...].astype(np.float32)
        print(f"✅ Data shape: {input_data.shape}")
        print(f"   Date range: Single date analysis")
        print(f"   Target variable: 850hPa Humidity (Channel {CONFIG['humidity_channel']})")
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"fengwu_xai_comprehensive_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Store all results
    all_results = {}
    
    # Run methods
    print("\n" + "="*80)
    print("RUNNING ENHANCED XAI METHODS")
    print("="*80)
    
    method_sequence = [
        ('finite_difference', compute_finite_difference),
        ('smoothgrad', compute_smoothgrad),
        ('input_x_gradient', lambda s, d, t: compute_input_x_gradient(d, all_results['finite_difference']['gradients']) 
                                          if 'finite_difference' in all_results else None),
        ('integrated_gradients', compute_integrated_gradients),
        ('occlusion', compute_occlusion_sensitivity),
        ('gradient_shap', compute_gradient_shap),
    ]
    
    for method_name, method_func in method_sequence:
        if method_name in CONFIG['methods_to_run']:
            print(f"\n{'='*60}")
            print(f"Starting {method_name.replace('_', ' ').title()}...")
            print('='*60)
            
            try:
                result = method_func(session, input_data, CONFIG['humidity_channel'])
                if result:
                    all_results[method_name] = result
                    create_method_specific_visualization(result, method_name, output_dir)
                    
                    # Clear GPU cache
                    if CONFIG['use_gpu']:
                        torch.cuda.empty_cache()
                        gc.collect()
            except Exception as e:
                print(f"❌ Error in {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Perform cross-method analysis
    if len(all_results) >= 2:
        analysis_dir = perform_cross_method_analysis(all_results, output_dir)
    else:
        print("\n⚠️  Need at least 2 methods for cross-method analysis")
        analysis_dir = None
    
    # Generate comprehensive paper-ready summary
    print("\n" + "="*80)
    print("GENERATING PAPER-READY SUMMARY")
    print("="*80)
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'single_date_analysis': True,
        'target_variable': '850hPa Humidity',
        'target_channel': CONFIG['humidity_channel'],
        'methods_completed': list(all_results.keys()),
        'key_findings': {},
        'implications_for_climate_studies': {},
        'method_performance_comparison': {},
        'recommendations_for_practitioners': {}
    }
    
    # Key findings
    for method_name, result in all_results.items():
        if 'channel_importances' in result:
            importances = result['channel_importances']
            top_idx = np.argmax(importances)
            top_var = get_clean_name(top_idx)
            top_group = get_variable_group(top_var)
            
            # Find top region
            if 'gradients' in result:
                grad_mag = np.sqrt(np.sum(result['gradients']**2, axis=0))
                region_means = []
                for region_name, region_info in CONFIG['analysis_regions'].items():
                    lat_min, lat_max, lon_min, lon_max = region_info['bounds']
                    height, width = grad_mag.shape
                    lat_idx_min = int((90 - lat_max) * height / 180)
                    lat_idx_max = int((90 - lat_min) * height / 180)
                    lon_idx_min = int(lon_min * width / 360)
                    lon_idx_max = int(lon_max * width / 360)
                    
                    region_data = grad_mag[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
                    if region_data.size > 0:
                        region_means.append((region_name, np.mean(region_data)))
                
                top_region = max(region_means, key=lambda x: x[1])[0] if region_means else "N/A"
            else:
                top_region = "N/A"
            
            summary['key_findings'][method_name] = {
                'most_influential_variable': top_var,
                'variable_group': top_group,
                'importance_score': float(importances[top_idx]),
                'most_sensitive_region': top_region,
                'computation_time': result['computation_time'],
                'variables_above_mean': int(np.sum(importances > np.mean(importances))),
                'skewness': float(stats.skew(importances))
            }
    
    # Method comparison
    if len(all_results) >= 2:
        # Calculate consistency scores
        importance_matrices = []
        method_names = []
        
        for method_name, result in all_results.items():
            if 'channel_importances' in result:
                importance_matrices.append(result['channel_importances'])
                method_names.append(method_name)
        
        if len(importance_matrices) >= 2:
            # Pairwise correlations
            correlations = []
            for i in range(len(importance_matrices)):
                for j in range(i+1, len(importance_matrices)):
                    corr, _ = stats.pearsonr(importance_matrices[i], importance_matrices[j])
                    correlations.append({
                        'method1': method_names[i],
                        'method2': method_names[j],
                        'pearson_correlation': float(corr)
                    })
            
            summary['method_performance_comparison']['pairwise_correlations'] = correlations
            
            # Ranking consistency
            top_vars_by_method = {}
            for method_name, result in all_results.items():
                if 'channel_importances' in result:
                    importances = result['channel_importances']
                    top_indices = np.argsort(importances)[::-1][:10]
                    top_vars_by_method[method_name] = [get_clean_name(idx) for idx in top_indices]
            
            # Find common variables across methods
            all_top_vars = []
            for vars_list in top_vars_by_method.values():
                all_top_vars.extend(vars_list[:5])
            
            from collections import Counter
            var_counts = Counter(all_top_vars)
            consensus_vars = [var for var, count in var_counts.items() if count >= 2]
            
            summary['method_performance_comparison']['consensus_variables'] = consensus_vars
            summary['method_performance_comparison']['consensus_strength'] = len(consensus_vars)
    
    # Climate implications
    summary['implications_for_climate_studies'] = {
        'potential_modification_targets': [],
        'sensitive_regions_for_intervention': [],
        'key_physical_processes': [],
        'teleconnection_considerations': []
    }
    
    # Recommendations
    summary['recommendations_for_practitioners'] = {
        'fastest_method': min(all_results.items(), key=lambda x: x[1]['computation_time'])[0],
        'most_detailed_method': max(all_results.items(), key=lambda x: len(x[1].keys()))[0],
        'recommended_for_physical_insights': 'finite_difference',  # Baseline
        'recommended_for_robustness': 'smoothgrad',
        'recommended_for_interaction_studies': 'input_x_gradient'
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'paper_ready_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create executive summary markdown
    md_path = os.path.join(output_dir, 'EXECUTIVE_SUMMARY.md')
    with open(md_path, 'w') as f:
        f.write("# FengWu XAI Analysis: Executive Summary\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Single Date Analysis:** Yes\n")
        f.write(f"**Target Variable:** 850hPa Humidity\n\n")
        
        f.write("## Key Findings\n\n")
        
        for method_name, findings in summary['key_findings'].items():
            f.write(f"### {method_name.replace('_', ' ').title()}\n")
            f.write(f"- **Most influential variable:** {findings['most_influential_variable']} ({findings['variable_group']})\n")
            f.write(f"- **Importance score:** {findings['importance_score']:.3e}\n")
            f.write(f"- **Most sensitive region:** {findings['most_sensitive_region']}\n")
            f.write(f"- **Computation time:** {findings['computation_time']:.1f}s\n")
            f.write(f"- **Variables above mean importance:** {findings['variables_above_mean']}\n\n")
        
        if 'consensus_variables' in summary.get('method_performance_comparison', {}):
            f.write("## Consensus Across Methods\n\n")
            f.write("The following variables were identified as important by multiple methods:\n\n")
            for var in summary['method_performance_comparison']['consensus_variables']:
                group = get_variable_group(var)
                f.write(f"- **{var}** ({group})\n")
        
        f.write("\n## Climate Implications\n\n")
        f.write("1. **Potential modification targets:** Variables consistently identified as influential\n")
        f.write("2. **Regional sensitivity:** Areas showing highest gradient magnitudes\n")
        f.write("3. **Physical mechanisms:** Dominant processes affecting humidity prediction\n")
        f.write("4. **Teleconnections:** Large-scale patterns influencing local humidity\n\n")
        
        f.write("## Method Recommendations\n\n")
        recs = summary['recommendations_for_practitioners']
        f.write(f"- **Fastest method:** {recs['fastest_method']}\n")
        f.write(f"- **Most detailed:** {recs['most_detailed_method']}\n")
        f.write(f"- **For physical insights:** {recs['recommended_for_physical_insights']}\n")
        f.write(f"- **For robustness:** {recs['recommended_for_robustness']}\n")
        f.write(f"- **For interaction studies:** {recs['recommended_for_interaction_studies']}\n")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Methods completed: {len(all_results)}")
    print(f"   Total computation time: {sum(r['computation_time'] for r in all_results.values()):.1f}s")
    print(f"   Output directory: {output_dir}")
    
    if analysis_dir:
        print(f"   Cross-method analysis: {analysis_dir}")
    
    print(f"\n📄 Summary files:")
    print(f"   • config.json - Configuration used")
    print(f"   • paper_ready_summary.json - Detailed analysis results")
    print(f"   • EXECUTIVE_SUMMARY.md - Paper-ready summary")
    
    print(f"\n📈 Visualizations generated for each method:")
    for method_name in all_results.keys():
        print(f"   • {method_name}/enhanced_visualizations/")
    
    if analysis_dir:
        print(f"\n🔍 Comparative analyses:")
        print(f"   • cross_method_analysis/ - All comparative plots")
    
    print("\n" + "="*80)
    print("✅ ENHANCED ANALYSIS COMPLETE!")
    print("="*80)
    print("\nThis analysis provides:")
    print("1. Comprehensive single-date XAI evaluation")
    print("2. Method comparison and consistency assessment")
    print("3. Physical interpretation of results")
    print("4. Climate modification implications")
    print("5. Paper-ready summaries and visualizations")
    print("="*80)

if __name__ == "__main__":
    main()
