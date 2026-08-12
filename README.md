# 🌦️ FengWu XAI Comparison Framework
## Multi-Method Explainable AI for AI based Weather Prediction Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 📋 Overview

The **FengWu XAI Comparison Framework** is a comprehensive explainability toolkit designed for the FengWu AI weather forecasting model. This framework implements and compares multiple state-of-the-art attribution methods to identify which atmospheric variables and regions most influence the model's predictions of 850hPa specific humidity.

This code accompanies our paper: *"Comparative Analysis of Explainable AI Methods for Exploring the Attribution of Specific Humidity Forecasts based on the FengWu AI Weather Forecasting Model"* (submitted to "Computers and Geosciences").

### 🎯 Key Features

- **6 XAI Methods** implemented: Finite Difference, SmoothGrad, Input×Gradient, Integrated Gradients, Occlusion Sensitivity, and Gradient SHAP
- **Multi-perspective Analysis**: Variable importance, regional sensitivity, physical mechanism analysis, and vertical profiling
- **Cross-method Comparison**: Correlation analysis, consensus identification, and statistical significance testing
- **Visualizations**: Publication-quality plots with consistent styling
- **Comprehensive Outputs**: JSON summaries, CSV exports, and Markdown reports
- **GPU Optimized**: CUDA support for fast computations (tested on NVIDIA A100 80GB)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/rsreza/Explanable-AI-for-Attribution-of-Weather-forecasts.git
cd Explanable-AI-for-Attribution-of-Weather-forecasts

# Install requirements mentioned below
pip install numpy>=1.21.0 torch>=2.0.0 onnxruntime>=1.14.0 matplotlib>=3.5.0 seaborn>=0.11.0 pandas>=1.3.0 cartopy>=0.20.0 scipy>=1.7.0 scikit-learn>=1.0.0
