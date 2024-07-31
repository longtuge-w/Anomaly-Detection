# Anomaly Detection in Wind Turbine Operations

## Introduction

This project aims to develop an early warning system for predicting or anticipating faults and failures in wind turbines. Utilizing data from EDP Renewables' wind farm, which includes five wind turbines over a two-year period, our goal is to leverage machine learning and time series analysis techniques to forecast critical maintenance events. This effort supports proactive maintenance strategies, reducing downtime and operational costs.

## Dataset

The dataset comprises SCADA signals from the turbines, meteorological data from the farm's met mast, and recorded failure events. It is divided into a 20-month training set and a 4-month test set, enabling the development and validation of predictive models.

## Code Files Overview

- `attn.py`: Implements the Anomaly Attention mechanism, defining custom attention layers to focus on significant patterns in the wind turbine data.
- `Dataset.py`: Contains the custom PyTorch dataset classes for loading and preprocessing signal and meteorological data.
- `embed.py`: Defines embedding layers used for encoding the input data, including positional and token embeddings for time series data.
- `loss.py`: Provides a custom loss function, `TotalSavingLoss`, designed to calculate the savings from correctly predicted maintenance events versus the costs of false predictions.
- `Model.py`: Constructs the Anomaly Transformer model, integrating attention mechanisms and embedding layers for anomaly detection.
- `utils.py`: Offers utility functions for data preprocessing, including standardization, handling missing values, and data sorting.

## Methodology

Our approach harnesses the power of Transformer-based models, adapted for time series data, to capture complex dependencies and temporal dynamics. By integrating custom attention mechanisms, we aim to highlight anomalous signals indicative of potential failures. The model architecture is designed to process both turbine-specific signals and environmental factors from meteorological data, acknowledging the multifaceted nature of turbine operation anomalies.

1. **Data Embedding**: Signal and meteorological data are first passed through embedding layers, enhancing the model's ability to interpret time series inputs.
2. **Anomaly Attention**: Utilizing a tailored attention mechanism, the model identifies patterns and dependencies that are characteristic of pre-failure states.
3. **Dual Input Encoder**: Combines embedded signal and meteorological data, allowing the model to consider a comprehensive view of factors affecting turbine health.
4. **Loss Calculation**: The `TotalSavingLoss` function aligns model training with the project's objective, focusing on maximizing the financial savings by accurately predicting maintenance needs.

## Conclusion

This project represents a significant step towards intelligent wind turbine maintenance. By predicting failures before they occur, we can substantially reduce unplanned downtime and maintenance costs, ensuring more reliable and efficient wind energy production.
