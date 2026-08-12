# 🌦️ FengWu XAI Comparison Framework
## Multi-Method Explainable AI for Weather Prediction Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 📋 Overview

The **FengWu XAI Comparison Framework** is a comprehensive explainability toolkit designed for the FengWu AI weather forecasting model. This framework implements and compares multiple state-of-the-art attribution methods to identify which atmospheric variables and regions most influence the model's predictions of 850hPa specific humidity.

This code accompanies our paper: "Comparative Analysis of Explainable AI Methods for Exploring the Attribution of Specific Humidity Forecasts based on the FengWu AI Weather Forecasting Model" (submitted to "Computers and Geosciences").

### 🎯 Key Features

- **6 XAI Methods** implemented: Finite Difference, SmoothGrad, Input×Gradient, Integrated Gradients, Occlusion Sensitivity, and Gradient SHAP
- **Multi-perspective Analysis**: Variable importance, regional sensitivity, physical mechanism analysis, and vertical profiling
- **Cross-method Comparison**: Correlation analysis, consensus identification, and statistical significance testing
- **Visualizations**: plots with consistent styling
- **Comprehensive Outputs**: JSON summaries, CSV exports, and Markdown reports
- **GPU Optimized**: CUDA support for fast computations


### Prerequisites

```bash
# Clone the repository
git clone https://github.com/rsreza/Explanable-AI-for-Attribution-of-Weather-forecasts.git
cd fengwu-xai-framework

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place your FengWu input files in the following structure:

```
input_data/
├── input1.npy  # First time step (69 channels)
└── input2.npy  # Second time step (69 channels)
```

Ensure the FengWu ONNX model file (`fengwu_v2.onnx`) is in the root directory.

### Run Analysis

```bash
# Basic execution
python fengwu_xai_comprehensive.py

# Custom configuration (edit CONFIG dict in the script)
```

### Output Structure

The framework generates a comprehensive output directory:

```
fengwu_xai_comprehensive_YYYYMMDD_HHMMSS/
├── config.json
├── paper_ready_summary.json
├── EXECUTIVE_SUMMARY.md
├── finite_difference/
│   ├── enhanced_visualizations/
│   │   ├── finite_difference_comprehensive_analysis.png
│   │   └── variable_importance.csv
│   └── data/
│       ├── gradients.npy
│       ├── gradient_magnitude.npy
│       └── channel_importances.npy
├── smoothgrad/
│   └── [similar structure]
├── input_x_gradient/
│   └── [similar structure]
├── integrated_gradients/
│   └── [similar structure]
├── occlusion/
│   └── [similar structure]
├── gradient_shap/
│   └── [similar structure]
└── cross_method_analysis/
    ├── cross_method_comparison.png
    ├── physical_consistency_analysis.png
    ├── regional_sensitivity_analysis.png
    ├── climate_implications.png
    ├── method_correlations.csv
    └── statistical_report.json
```

---

## 📊 XAI Methods Implemented

| Method | Description | Citation |
|--------|-------------|----------|
| **Finite Difference** | Basic gradient computation via input perturbation | Baseline method |
| **SmoothGrad** | Noise-averaged gradients for robustness | Smilkov et al., 2017 |
| **Input×Gradient** | Input-scaled gradients showing interaction effects | Shrikumar et al., 2016 |
| **Integrated Gradients** | Path-integrated attribution method | Sundararajan et al., 2017 |
| **Occlusion Sensitivity** | Patch-based perturbation analysis | Zeiler & Fergus, 2014 |
| **Gradient SHAP** | SHAP-like attribution using multiple baselines | Lundberg & Lee, 2017 |

---

## 🔬 Analysis Capabilities

### 1. Variable Importance Ranking
- Identifies top influential variables from 138 input channels
- Groups by physical categories (Temperature, Wind, Geopotential, Moisture, Surface)
- Provides ranked CSV exports with normalized importance scores

### 2. Regional Sensitivity Analysis
- Predefined climate regions: Iran, Mediterranean, Arabian Sea, Caspian Sea, Persian Gulf, Indian Subcontinent, North Africa
- Customizable region definitions
- Comparative analysis across methods

### 3. Physical Mechanism Analysis
- **Thermodynamic**: Temperature-driven processes
- **Dynamic**: Wind and pressure systems
- **Moisture**: Water vapor transport
- **Upper-level**: Atmospheric dynamics at altitude

### 4. Vertical Profiling
- Importance distribution across pressure levels (50-1000 hPa)
- Method comparison of vertical sensitivity patterns

### 5. Cross-Method Statistics
- Pearson correlation between method rankings
- Spearman rank correlation
- KL divergence for importance distributions
- Jaccard similarity for top-N variables

### 6. Teleconnection Analysis
- Sensitivity patterns for Monsoon, Mediterranean, Subtropical, and Polar regions
- Large-scale circulation implications

---

## 📈 Visual Outputs

### Per-Method Visualizations
- **Global Saliency Map**: Spatial distribution of importance
- **Top Variable Patterns**: Gradient maps for most influential variables
- **Variable Importance Ranking**: Bar chart with color-coded physical groups
- **Regional Sensitivity**: Comparative bar chart
- **Uncertainty Maps** (SmoothGrad): Gradient stability visualization
- **Signed Importance Distribution**: Histogram of positive/negative contributions

### Cross-Method Visualizations
- **Method Correlation Heatmap**: Spearman/Pearson correlation matrices
- **Top Variable Consistency**: Jaccard similarity matrix
- **Computational Efficiency**: Performance comparison
- **Physical Group Distribution**: Method comparison by variable categories
- **Regional Sensitivity Matrix**: Heatmap of regional responses
- **PCA Visualization**: Method similarity analysis

---

## 📝 Configuration

The `CONFIG` dictionary allows customization of all analysis parameters:

```python
CONFIG = {
    'use_gpu': True,                    # Enable GPU acceleration
    'gpu_device_id': 0,                 # GPU device selection
    'humidity_channel': 137,            # Target channel (850hPa humidity)
    'epsilon': 1e-4,                    # Perturbation magnitude
    
    'methods_to_run': [                 # Select specific methods
        'finite_difference',
        'smoothgrad',
        'input_x_gradient',
        'integrated_gradients',
        'occlusion',
        'gradient_shap'
    ],
    
    'smoothgrad_samples': 5,            # Number of noise samples
    'integrated_gradients_steps': 20,   # Integration steps
    'occlusion_window_size': (3, 3),    # Occlusion patch size
    
    'top_n_variables': 15,              # Variables to display
    'correlation_threshold': 0.7,       # Significance threshold
    
    'analysis_regions': {               # Define custom regions
        'Region_Name': {'bounds': (lat_min, lat_max, lon_min, lon_max), 'color': 'red'}
    },
    
    'variable_groups': {                # Physical variable grouping
        'Temperature': ['t2m', 't50', ...],
        'Wind': ['u10', 'v10', ...],
        # ...
    }
}
```

---

## 🔧 Requirements

```
python >= 3.8
numpy >= 1.21.0
torch >= 2.0.0
onnxruntime >= 1.14.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
pandas >= 1.3.0
cartopy >= 0.20.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
```

---


### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black fengwu_xai_comprehensive.py

# Type checking
mypy fengwu_xai_comprehensive.py
```

---

## 📖 Documentation

### Understanding the Output

#### Variable Importance CSV
| Rank | Channel | Variable | Group | Importance | Normalized_Importance | Time_Step |
|------|---------|----------|-------|------------|----------------------|-----------|
| 1 | 137 | q850 | Moisture | 2.34e-3 | 0.156 | input2 |
| 2 | 93 | z850 | Geopotential | 1.87e-3 | 0.125 | input2 |

#### Statistical Report
- Method-wise statistics (mean, std, skewness, kurtosis)
- Pairwise comparisons (correlation coefficients, p-values)
- KL divergence for method similarity assessment
- Top variable consistency metrics

### Interpreting Results

1. **High Importance Score**: Variable strongly influences humidity prediction
2. **Positive/Negative Gradient**: Directional influence on predicted humidity
3. **Regional Sensitivity**: Areas where model is most responsive
4. **Method Consensus**: Variables identified by multiple methods = robust importance

---

## 🧪 Validation

The framework includes validation mechanisms:
- Sensitivity to perturbation magnitude (`epsilon`)
- Convergence checks for integrated gradients
- Uncertainty quantification via SmoothGrad variance
- Statistical significance testing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- The FengWu team for the weather forecasting model
- Our research collaborators at [Institution Name]
- Funding sources: [Grant Information]

---

## 📧 Contact

For questions, issues, or collaborations:
- **Lead Author**: [Your Name] ([email@domain.com](mailto:email@domain.com))
- **GitHub Issues**: [Issues Page](https://github.com/yourusername/fengwu-xai-framework/issues)
- **Paper Preprint**: [Link to arXiv or preprint]

---

## 📌 Quick Links

- [Documentation](docs/)
- [Examples](examples/)
- [FAQ](FAQ.md)
- [Changelog](CHANGELOG.md)
- [Paper Preprint](https://arxiv.org/abs/XXXX.XXXXX)

---

## ⭐ Star History

If you find this framework useful, please consider starring the repository! ⭐

---

**Made with ❤️ for climate science and AI interpretability**

---

### Additional Files You Might Want:

#### requirements.txt
```txt
numpy>=1.21.0
torch>=2.0.0
onnxruntime>=1.14.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
cartopy>=0.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

#### CITATION.cff
```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "YourName"
    given-names: "YourFirstName"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "FengWu XAI Comparison Framework"
version: 1.0.0
date-released: 2026-08-12
url: "https://github.com/yourusername/fengwu-xai-framework"
```

#### CONTRIBUTING.md
```markdown
# Contributing to FengWu XAI Framework

We love your input! We want to make contributing to this project as easy and transparent as possible.

## Development Process
1. Fork the repo and create your branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Submit a pull request

## Pull Request Guidelines
- Update documentation for any new features
- Add tests for new functionality
- Follow existing code style
- Reference relevant issues in PR description
```

This README provides a professional, comprehensive overview that will make your GitHub repository attractive to reviewers, readers, and other researchers who might want to reproduce or build upon your work.
