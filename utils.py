import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from Dataset import *


# Create a dictionary to map component names to column names
component_columns = {
    'GENERATOR': 'Generator_Failure_Days',
    'HYDRAULIC_GROUP': 'Hydraulic_Failure_Days',
    'GENERATOR_BEARING': 'Generator_Bearing_Failure_Days',
    'TRANSFORMER': 'Transformer_Failure_Days',
    'GEARBOX': 'Gearbox_Failure_Days'
}


def sort_dataframe(df, sort_by_turbine_id=False):
    """
    Sorts the DataFrame by 'Timestamp' and optionally by 'Turbine_ID'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to sort.
    sort_by_turbine_id (bool): Whether to also sort by 'Turbine_ID'.
    
    Returns:
    pd.DataFrame: The sorted DataFrame.
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    if sort_by_turbine_id:
        df.sort_values(by=['Turbine_ID', 'Timestamp'], inplace=True)
    else:
        df.sort_values(by=['Timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fill_nan(df, group_by_turbine_id=False):
    """
    Fills missing values using forward fill. If 'group_by_turbine_id' is True,
    it performs the operation within each 'Turbine_ID' group.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to fill missing values in.
    group_by_turbine_id (bool): Whether to group by 'Turbine_ID' before filling.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    df = df.copy()
    if group_by_turbine_id:
        df_filled = df.groupby('Turbine_ID').transform(lambda x: x.ffill())
        df_filled.insert(loc=0, column='Turbine_ID', value=df['Turbine_ID'])
    else:
        df_filled = df.ffill()
    return df_filled


def standardize_features(df, identifier_columns):
    """
    Standardizes the feature columns of the DataFrame using z-score.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with features to standardize.
    identifier_columns (list): List of columns to exclude from standardization.
    
    Returns:
    pd.DataFrame: The DataFrame with standardized feature columns.
    """
    df = df.copy()
    feature_columns = [col for col in df.columns if col not in identifier_columns]
    features_array = df[feature_columns].to_numpy()
    standardized_features = (features_array - np.mean(features_array, axis=1, keepdims=True)) / np.std(features_array, axis=1, keepdims=True)
    standardized_features = np.nan_to_num(standardized_features)
    df[feature_columns] = standardized_features
    return df


def preprocess_data(signals, metmast):
    """
    Preprocesses the 'signals' and 'metmast' DataFrames by sorting, filling NaN values,
    and standardizing feature columns.

    Parameters:
    signals (pd.DataFrame): The signals DataFrame to preprocess.
    metmast (pd.DataFrame): The metmast DataFrame to preprocess.

    Returns:
    tuple: A tuple containing the preprocessed 'signals' and 'metmast' DataFrames.
    """
    # Sort the DataFrames
    signals_sorted = sort_dataframe(signals, sort_by_turbine_id=True)
    metmast_sorted = sort_dataframe(metmast)

    # Fill NaN values
    signals_filled = fill_nan(signals_sorted, group_by_turbine_id=True)
    metmast_filled = fill_nan(metmast_sorted)

    # Standardize features
    signals_standardized = standardize_features(signals_filled, ['Turbine_ID', 'Timestamp'])
    metmast_standardized = standardize_features(metmast_filled, ['Timestamp'])

    return signals_standardized, metmast_standardized


def create_month_one_hot_encoding(data):
    # Assuming 'Timestamp' is the column name for the timestamp data
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Extract the month from the 'Timestamp' column
    data['Month'] = data['Timestamp'].dt.month
    
    # Create one-hot encoding for the 'Month' column
    month_one_hot = pd.get_dummies(data['Month'], prefix='Month').astype(np.int32)
    
    # Concatenate the one-hot encoded features with the original data
    data_with_month_one_hot = pd.concat([data, month_one_hot], axis=1)
    
    # Drop the 'Month' column since it's no longer needed
    data_with_month_one_hot.drop('Month', axis=1, inplace=True)
    
    return data_with_month_one_hot


def average_duplicates(df, group_cols=['Turbine_ID', 'Timestamp']):
    """
    Averages duplicate rows in a DataFrame based on specified grouping columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    group_cols (list): The columns to group by for averaging (default: ['Turbine_ID', 'Timestamp']).

    Returns:
    pd.DataFrame: The DataFrame with duplicate rows averaged.
    """
    # Ensure 'Timestamp' is in the proper datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Group by the specified columns and calculate the mean for each group
    # This will automatically handle numeric columns and ignore non-numeric ones
    averaged_df = df.groupby(group_cols).mean().reset_index()

    return averaged_df


def merge_and_fill(signals, metmast):
    """
    Merges the 'signals' and 'metmast' DataFrames on 'Timestamp' and fills missing values in 'metmast'.

    Parameters:
    signals (pd.DataFrame): The DataFrame containing signal data.
    metmast (pd.DataFrame): The DataFrame containing metmast data.

    Returns:
    tuple: A tuple containing the merged 'signals' and 'metmast' DataFrames.
    """
    # Convert the 'Timestamp' column to datetime in both DataFrames
    signals['Timestamp'] = pd.to_datetime(signals['Timestamp'])
    metmast['Timestamp'] = pd.to_datetime(metmast['Timestamp'])

    # Create a DataFrame with unique timestamps from both DataFrames
    unique_timestamps = pd.Series(pd.unique(signals['Timestamp'])).to_frame(name='Timestamp')

    # Merge the two DataFrames on 'Timestamp', retaining all timestamps
    metmast = pd.merge(unique_timestamps, metmast, on='Timestamp', how='left').sort_values(by='Timestamp')

    # Forward fill the NaN values for the second DataFrame's columns
    second_df_columns = metmast.columns.drop('Timestamp')
    metmast[second_df_columns] = metmast[second_df_columns].ffill()

    return signals, metmast


def fill_missing_signal_data(df):
    """
    Fills missing signal data in a DataFrame by creating a complete DataFrame with all combinations of IDs and timestamps.

    Parameters:
    df (pd.DataFrame): The DataFrame to fill missing data in.

    Returns:
    pd.DataFrame: The DataFrame with missing data filled.
    """
    # Convert 'Timestamp' to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Get unique timestamps and IDs
    unique_timestamps = np.sort(pd.unique(df['Timestamp']))
    unique_ids = np.sort(pd.unique(df['Turbine_ID']))

    # Create a complete DataFrame with all combinations of IDs and timestamps
    complete_df = pd.MultiIndex.from_product([unique_ids, unique_timestamps], names=['Turbine_ID', 'Timestamp']).to_frame(index=False)

    # Merge the complete DataFrame with the original to ensure all combinations are present
    df_full = pd.merge(complete_df, df, on=['Turbine_ID', 'Timestamp'], how='left').sort_values(by=['Turbine_ID', 'Timestamp'])

    # Forward fill NaN values for each ID without setting 'Turbine_ID' as index
    turbine_id = df_full['Turbine_ID'].copy()
    df_filled = df_full.groupby('Turbine_ID', as_index=False).ffill()
    df_filled.insert(0, "Turbine_ID", turbine_id)

    return df_filled


def process_signal_data(signal_data, failure_data, window_size):
    signal_data, failure_data = signal_data.copy(), failure_data.copy()
    # Convert the timestamp columns to datetime
    signal_data['Timestamp'] = pd.to_datetime(signal_data['Timestamp'])
    failure_data['Timestamp'] = pd.to_datetime(failure_data['Timestamp'])
    
    # Initialize new columns with -1 (no failure)
    for col in component_columns.values():
        signal_data[col] = -1
    
    # Iterate over each turbine
    for turbine_id in signal_data['Turbine_ID'].unique():
        # Get the signal data and failure data for the current turbine
        turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id]
        turbine_failure_data = failure_data[failure_data['Turbine_ID'] == turbine_id]
        
        # Iterate over each component
        for component, column in component_columns.items():
            # Get the failure timestamps for the current component
            component_failures = turbine_failure_data[turbine_failure_data['Component'] == component]['Timestamp']
            
            # Iterate over each failure timestamp
            for failure_timestamp in component_failures:
                # Find the signal data within the window_size before the failure timestamp
                mask = (turbine_signal_data['Timestamp'] >= failure_timestamp - pd.Timedelta(days=window_size)) & (turbine_signal_data['Timestamp'] < failure_timestamp)
                signal_within_window = turbine_signal_data.loc[mask]
                
                # Calculate the number of days until the failure for data points within the window
                days_to_failure = (failure_timestamp - signal_within_window['Timestamp']).dt.days
                
                # Update the corresponding column with the number of days to failure only for data points within the window
                signal_data.loc[signal_within_window.index, column] = days_to_failure
    
    # Sort the signal data by timestamp
    signal_data = signal_data.sort_values(['Turbine_ID', 'Timestamp'])
    
    return signal_data


def get_all_feature_label(signal_data, component='all', window_size=60, method='classification'):

    signal_data = signal_data.copy()
    
    # Get the relevant features
    features = signal_data.columns.drop(['Turbine_ID', 'Timestamp'] + list(component_columns.values()))
    
    # Create an empty list to store the preprocessed data for each turbine
    preprocessed_data = []
    
    # Iterate over each turbine
    for turbine_id in signal_data['Turbine_ID'].unique():
        # Get the signal data for the current turbine
        turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id]
        
        # Convert the signal data to numpy array
        X = turbine_signal_data[features].values
        
        # Create a sliding window view of the data
        X_windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(window_size, X.shape[1]))
        
        if component == 'all':
            # Get the failure days for each component
            failure_days = turbine_signal_data[list(component_columns.values())].values[window_size-1:]
        else:
            # Get the failure days for each component
            failure_days = turbine_signal_data[component_columns[component]].values[window_size-1:]

        if method == 'classification':
            # Set labels between 1 and 60 and all other labels to 0
            failure_days = np.where((failure_days >= 1) & (failure_days <= 60), 1, 0)
        elif method == 'regression':
            # Set labels between 1 and 60 and all other labels to 0
            failure_days = np.where((failure_days >= 1) & (failure_days <= 60), failure_days, 0)
        else:
            raise ValueError(f"The parameter method should be classification/regression, get {method} instead")
        
        # Append the preprocessed data for the current turbine to the list
        preprocessed_data.append((X_windows, failure_days))
    
    # Concatenate the preprocessed data from all turbines
    X_tensor = np.concatenate([data[0] for data in preprocessed_data], axis=0)
    y_tensor = np.concatenate([data[1] for data in preprocessed_data], axis=0)
    
    # Remove the second dimension (with size 1) from X_tensor
    X_tensor = np.squeeze(X_tensor, axis=1)
    
    return X_tensor, y_tensor


def get_ind_feature_label(signal_data, turbine_id, component, window_size=60, method='classification'):

    signal_data = signal_data.copy()
    
    # Get the relevant features
    features = signal_data.columns.drop(['Turbine_ID', 'Timestamp'] + list(component_columns.values()))
    
    # Get the signal data for the current turbine
    turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id]
    
    # Convert the signal data to numpy array
    X = turbine_signal_data[features].values
    
    # Create a sliding window view of the data
    X_windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(window_size, X.shape[1]))
    
    # Get the failure days for each component
    failure_days = turbine_signal_data[component_columns[component]].values[window_size-1:]

    if method == 'classification':
        # Set labels between 1 and 60 and all other labels to 0
        failure_days = np.where((failure_days >= 1) & (failure_days <= 60), 1, 0)
    elif method == 'regression':
        # Set labels between 1 and 60 and all other labels to 0
        failure_days = np.where((failure_days >= 1) & (failure_days <= 60), failure_days, 0)
    else:
        raise ValueError(f"The parameter method should be classification/regression, get {method} instead")
    
    # Remove the second dimension (with size 1) from X_tensor
    X_tensor = np.squeeze(X_windows, axis=1)
    y_tensor = failure_days
    
    return X_tensor, y_tensor


def process_metmast_data(metmast_data, window_size=60, repeat=5):

    metmast_data = metmast_data.copy()

    # Convert the timestamp column to datetime
    metmast_data['Timestamp'] = pd.to_datetime(metmast_data['Timestamp'])
    
    # Sort the metmast data by timestamp
    metmast_data = metmast_data.sort_values('Timestamp')
    
    # Get the relevant features
    features = metmast_data.columns.drop('Timestamp')
    
    # Convert the metmast data to numpy array
    X_metmast = metmast_data[features].values
    
    # Create a sliding window view of the data
    X_metmast_windows = np.lib.stride_tricks.sliding_window_view(X_metmast, window_shape=(window_size, X_metmast.shape[1]))
    
    # Remove the second dimension (with size 1) from X_metmast_windows if it exists
    if X_metmast_windows.ndim > 3:
        X_metmast_windows = np.squeeze(X_metmast_windows, axis=1)
    
    if repeat > 1:
        # Repeat each window 5 times along a new axis
        X_metmast_windows_repeated = np.repeat(X_metmast_windows[:, np.newaxis, :, :], repeats=5, axis=1)
        # Reshape to [5B, T, F] by moving the repeat axis to the first position and reshaping
        B, repeats, T, F = X_metmast_windows_repeated.shape
        X_metmast_windows_repeated = X_metmast_windows_repeated.reshape(B * repeats, T, F)
    else:
        X_metmast_windows_repeated = X_metmast_windows
        
    return X_metmast_windows_repeated


def split_data(X_signal, y_signal, X_metmast, num_parts=5, train_ratio=0.8):
    
    total_samples = X_signal.shape[0]
    part_size = total_samples // num_parts
    remainder = total_samples % num_parts
    
    train_signal_indices = []
    val_signal_indices = []
    
    for i in range(num_parts):
        start_idx = i * part_size
        # Adjust the end index for the last part to include the remainder
        end_idx = (i + 1) * part_size if i < num_parts - 1 else total_samples
        part_indices = list(range(start_idx, end_idx))
        train_part_size = int(train_ratio * len(part_indices))
        train_signal_indices.extend(part_indices[:train_part_size])
        val_signal_indices.extend(part_indices[train_part_size:])
    
    X_train_signal = X_signal[train_signal_indices]
    y_train_signal = y_signal[train_signal_indices]
    X_val_signal = X_signal[val_signal_indices]
    y_val_signal = y_signal[val_signal_indices]
    
    X_train_metmast = X_metmast[train_signal_indices]
    X_val_metmast = X_metmast[val_signal_indices]
    
    return X_train_signal, y_train_signal, X_val_signal, y_val_signal, X_train_metmast, X_val_metmast


def create_labels_dataframe_all(signal_data, labels, window_size):

    # Create an empty list to store the preprocessed data for each turbine
    data = []
    
    # Iterate over each turbine
    for turbine_id in signal_data['Turbine_ID'].unique():
        # Get the signal data for the current turbine
        turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id][['Turbine_ID', 'Timestamp']]
        turbine_signal_data = turbine_signal_data.iloc[window_size-1:,:]
        data.append(turbine_signal_data)

    prediction_df = pd.concat(data, axis=0).reset_index(drop=True)
    prediction_df['Label'] = labels

    return prediction_df


def plot_all_labels(df):
    # Get unique turbine IDs
    turbine_ids = df['Turbine_ID'].unique()
    
    # Convert the 'Timestamp' column to datetime if it's not already
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Iterate over each turbine ID
    for turbine_id in turbine_ids:
        # Filter the DataFrame based on the turbine ID
        turbine_data = df[df['Turbine_ID'] == turbine_id].copy()
        
        # Set the 'Timestamp' column as the index
        turbine_data.set_index('Timestamp', inplace=True)
        
        # Create a new figure and axis for each turbine
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the prediction and label as lines
        ax.plot(turbine_data.index, turbine_data['Label'], label='Label')
        
        # Set the x-axis to display in monthly frequency
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Set the x-axis and y-axis labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        # Set the title of the plot
        ax.set_title(f'Turbine {turbine_id} - Prediction vs Label')
        
        # Add a legend
        ax.legend()
        
        # Adjust the layout to prevent overlapping of labels
        plt.tight_layout()
        
        # Display the plot
        plt.show()


def train_model(model, train_signal_dataset, train_metmast_dataset, val_signal_dataset, val_metmast_dataset, desc, optimizer, criterion, num_epochs, batch_size, device, clip_value=1.0, patience=10):
    
    if not os.path.exists("Model/"):
        os.makedirs("Model/")

    model_file_path = f"Model/{desc}.pt"
    
    if os.path.exists(model_file_path):
        print(f"Loading existing model from {model_file_path}")
        model.load_state_dict(torch.load(model_file_path))
    else:
        print(f"Training new model and saving to {model_file_path}")
    
    train_signal_loader = DataLoader(train_signal_dataset, batch_size=batch_size, shuffle=True)
    train_metmast_loader = DataLoader(train_metmast_dataset, batch_size=batch_size, shuffle=True)
    val_signal_loader = DataLoader(val_signal_dataset, batch_size=batch_size, shuffle=False)
    val_metmast_loader = DataLoader(val_metmast_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(zip(train_signal_loader, train_metmast_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", total=len(train_signal_loader))
        # progress_bar = tqdm(zip(train_signal_loader, train_metmast_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", total=5)
        # batch_counter = 0  # Initialize a counter for the batches

        for (signal_batch, label_batch), metmast_batch in progress_bar:
            # if batch_counter >= 5:  # Break after processing 5 batches
            #     break
            # batch_counter += 1
            
            signal_batch = signal_batch.to(device).float()
            metmast_batch = metmast_batch.to(device).float()
            label_batch = label_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(signal_batch, metmast_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        train_loss /= len(train_signal_loader)

        val_loss = evaluate_model(model, val_signal_loader, val_metmast_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f'EarlyStopping counter: {epochs_without_improvement} out of {patience}')

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), model_file_path)

    return model


def evaluate_model(model, signal_loader, metmast_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    
    # Initialize the progress bar
    progress_bar = tqdm(zip(signal_loader, metmast_loader), desc="Evaluating", unit="batch", total=len(signal_loader))
    
    with torch.no_grad():
        for (signal_batch, label_batch), metmast_batch in progress_bar:
            signal_batch = signal_batch.to(device).float()
            metmast_batch = metmast_batch.to(device).float()
            label_batch = label_batch.to(device).float()

            outputs = model(signal_batch, metmast_batch)
            loss = criterion(outputs, label_batch)
            val_loss += loss.item()
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix({"Loss": loss.item()})

    val_loss /= len(signal_loader)
    return val_loss


def get_ind_feature(signal_data, turbine_id, window_size=60):
    
    # Get the relevant features
    features = signal_data.columns.drop(['Turbine_ID', 'Timestamp'])
    
    # Get the signal data for the current turbine
    turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id]
    
    # Convert the signal data to numpy array
    X = turbine_signal_data[features].values
    
    # Create a sliding window view of the data
    X_windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(window_size, X.shape[1]))
    
    # Remove the second dimension (with size 1) from X_tensor
    X_tensor = np.squeeze(X_windows, axis=1)
    
    return X_tensor


def get_all_feature(signal_data, window_size=60):

    signal_data = signal_data.copy()
    
    # Get the relevant features
    features = signal_data.columns.drop(['Turbine_ID', 'Timestamp'])
    
    # Create an empty list to store the preprocessed data for each turbine
    preprocessed_data = []
    
    # Iterate over each turbine
    for turbine_id in signal_data['Turbine_ID'].unique():
        # Get the signal data for the current turbine
        turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id]
        
        # Convert the signal data to numpy array
        X = turbine_signal_data[features].values
        
        # Create a sliding window view of the data
        X_windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(window_size, X.shape[1]))

        preprocessed_data.append(X_windows)
    
    # Concatenate the preprocessed data from all turbines
    X_tensor = np.concatenate(preprocessed_data, axis=0)
    
    # Remove the second dimension (with size 1) from X_tensor
    X_tensor = np.squeeze(X_tensor, axis=1)
    
    return X_tensor


def preprocess_test_data(train_signal_data, train_metmast_data, test_signal_data, test_metmast_data, turbine_id, window_size=144):

    train_signal_data, train_metmast_data = train_signal_data.copy(), train_metmast_data.copy()
    test_signal_data, test_metmast_data = test_signal_data.copy(), test_metmast_data.copy()

    # Preprocess test signal data
    test_signal_data = average_duplicates(test_signal_data)
    test_signal_data = fill_missing_signal_data(test_signal_data)
    test_signal_data, test_metmast_data = merge_and_fill(test_signal_data, test_metmast_data)
    test_signals_filled, test_metmast_filled = preprocess_data(test_signal_data, test_metmast_data)
    # test_metmast_filled = create_month_one_hot_encoding(test_metmast_filled)

    # Concatenate training and test data
    concatenated_signal_data = pd.concat([train_signal_data, test_signals_filled], ignore_index=True)
    concatenated_metmast_data = pd.concat([train_metmast_data, test_metmast_filled], ignore_index=True)

    # Sort the DataFrames
    signals_sorted = sort_dataframe(concatenated_signal_data, sort_by_turbine_id=True)
    metmast_sorted = sort_dataframe(concatenated_metmast_data)

    # Get features and labels for the concatenated data
    X_signal = get_ind_feature(signals_sorted, turbine_id=turbine_id, window_size=window_size)
    X_metmast = process_metmast_data(metmast_sorted, window_size=window_size, repeat=1)

    # Get the test data from the concatenated data
    test_size = len(test_metmast_data)
    X_test_signal = X_signal[-test_size:]
    X_test_metmast = X_metmast[-test_size:]

    return X_test_signal, X_test_metmast, test_signals_filled, test_metmast_filled


def preprocess_test_data_all(train_signal_data, train_metmast_data, test_signal_data, test_metmast_data, window_size=144):

    train_signal_data, train_metmast_data = train_signal_data.copy(), train_metmast_data.copy()
    test_signal_data, test_metmast_data = test_signal_data.copy(), test_metmast_data.copy()

    # Preprocess test signal data
    test_signal_data = average_duplicates(test_signal_data)
    test_signal_data = fill_missing_signal_data(test_signal_data)
    test_signal_data, test_metmast_data = merge_and_fill(test_signal_data, test_metmast_data)
    test_signals_filled, test_metmast_filled = preprocess_data(test_signal_data, test_metmast_data)
    # test_metmast_filled = create_month_one_hot_encoding(test_metmast_filled)

    # Concatenate training and test data
    concatenated_signal_data = pd.concat([train_signal_data, test_signals_filled], ignore_index=True)
    concatenated_metmast_data = pd.concat([train_metmast_data, test_metmast_filled], ignore_index=True)

    # Sort the DataFrames
    signals_sorted = sort_dataframe(concatenated_signal_data, sort_by_turbine_id=True)
    metmast_sorted = sort_dataframe(concatenated_metmast_data)

    # Get features and labels for the concatenated data
    X_signal = get_all_feature(signals_sorted, window_size=window_size)
    X_metmast = process_metmast_data(metmast_sorted, window_size=window_size)

    # Get the test data from the concatenated data
    test_size = len(test_signal_data)
    X_test_signal = X_signal[-test_size:]
    X_test_metmast = X_metmast[-test_size:]

    return X_test_signal, X_test_metmast, test_signals_filled, test_metmast_filled


def get_predictions(model, X_test_signal, X_test_metmast, desc, device):
    model_file_path = f"Model/{desc}.pt"
    print(f"Loading existing model from {model_file_path}")
    model.load_state_dict(torch.load(model_file_path), strict=True)
    model.eval()

    test_signal_dataset = SignalDataset(torch.from_numpy(X_test_signal), torch.zeros(X_test_signal.shape[0]))
    test_metmast_dataset = MetmastDataset(torch.from_numpy(X_test_metmast))

    test_signal_loader = DataLoader(test_signal_dataset, batch_size=1024, shuffle=False)
    test_metmast_loader = DataLoader(test_metmast_dataset, batch_size=1024, shuffle=False)

    predictions = []

    with torch.no_grad():
        progress_bar = tqdm(zip(test_signal_loader, test_metmast_loader), total=len(test_signal_loader), desc="Processing Batches")
        for (signal_batch, _), metmast_batch in progress_bar:
            signal_batch = signal_batch.to(device).float()
            metmast_batch = metmast_batch.to(device).float()

            outputs = model(signal_batch, metmast_batch)
            predictions.extend(outputs.cpu().numpy())

    return predictions


def create_prediction_dataframe(test_metmast_data, predictions, turbine_id):
    # Get the turbine IDs and timestamps from the test signal data
    timestamps = np.sort(pd.unique(test_metmast_data['Timestamp']))

    # Create a DataFrame with turbine IDs, timestamps, and predictions
    prediction_data = {
        'Turbine_ID': turbine_id,
        'Timestamp': timestamps,
        'Prediction': predictions
    }
    prediction_df = pd.DataFrame(prediction_data)

    return prediction_df


def create_prediction_dataframe_all(signal_data, predictions, window_size):

    # Create an empty list to store the preprocessed data for each turbine
    data = []
    
    # Iterate over each turbine
    for turbine_id in signal_data['Turbine_ID'].unique():
        # Get the signal data for the current turbine
        turbine_signal_data = signal_data[signal_data['Turbine_ID'] == turbine_id][['Turbine_ID', 'Timestamp']]
        turbine_signal_data = turbine_signal_data.iloc[window_size-1:,:]
        data.append(turbine_signal_data)

    prediction_df = pd.concat(data, axis=0).reset_index(drop=True)
    prediction_df['Prediction'] = predictions

    return prediction_df