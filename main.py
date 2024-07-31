# Import necessary modules and models from different files.
from utils import *
from Dataset import SignalDataset, MetmastDataset
from LSTM import DualInputLSTM
from ALSTM import DualInputALSTM
from TCN import DualInputTCNModel
from SFM import DualInputSFMModel
from GATS import DualInputGATModel

# Set the device to CUDA if a GPU is available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main block to ensure the script is run directly and not imported.
if __name__ == "__main__":

    # Initialize configuration parameters for the dataset and model training.
    turbine_id = 'T06'
    component = 'GENERATOR'
    method = 'classification'
    data_type = 'ind' # 'all' for all data, 'ind' for individual data.
    model_name = 'LSTM'
    loss_name = 'BCE'
    window_size = 24 * 6  # Window size for time-series data processing.

    # Constants for model dimensions and training configuration.
    SIGNAL_SIZE = 81
    METMAST_SIZE = 52
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    NUM_LAYERS = 5
    DROPOUT = 0.1
    NUM_EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-4
    CLIP_VALUE = 5.0  # For gradient clipping to avoid exploding gradients.
    PATIENCE = 10  # For early stopping if validation loss does not improve.

    # Load datasets from CSV files.
    print("Loading datasets ...")
    signals = pd.read_csv("wind-farm-1-signals-training.csv", sep=';')
    metmast = pd.read_csv("wind-farm-1-metmast-training.csv", sep=';')
    failures = pd.read_csv("htw-failures-training.csv")

    # Preprocess the datasets for model training.
    print("Preprocessing data ...")
    signals = average_duplicates(signals)
    signals = fill_missing_signal_data(signals)
    signals, metmast = merge_and_fill(signals, metmast)
    signals_filled, metmast_filled = preprocess_data(signals, metmast)

    # Prepare processed datasets for training and validation.
    print("Preparing datasets ...")
    signal_processed = process_signal_data(signals_filled, failures)
    X_metmast = process_metmast_data(metmast_filled)

    # Extract features and labels according to the specified data_type.
    if data_type == 'all':
        X_signal, y_signal = get_all_feature_label(signal_processed, turbine_id, component, window_size=window_size, method=method)
    elif data_type == 'ind':
        X_signal, y_signal = get_ind_feature_label(signal_processed, turbine_id, component, window_size=window_size, method=method)
    else:
        raise ValueError("Parameter data_type should be 'all'/'ind', got '{}' instead".format(data_type))

    # Split the processed data into training and validation sets.
    X_train_signal, y_train_signal, X_val_signal, y_val_signal, X_train_metmast, X_val_metmast = split_data(X_signal, y_signal, X_metmast)

    # Create dataset instances for training and validation.
    train_signal_dataset = SignalDataset(X_train_signal, y_train_signal)
    val_signal_dataset = SignalDataset(X_val_signal, y_val_signal)
    train_metmast_dataset = MetmastDataset(X_train_metmast)
    val_metmast_dataset = MetmastDataset(X_val_metmast)

    # Instantiate the model based on the model_name parameter.
    if model_name == 'LSTM':
        model = DualInputLSTM(
            signal_input_size=SIGNAL_SIZE,
            metmast_input_size=METMAST_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)
    elif model_name == 'ALSTM':
        model = DualInputALSTM(
            signal_input_size=SIGNAL_SIZE,
            metmast_input_size=METMAST_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)
    elif model_name == 'TCN':
        model = DualInputTCNModel(
            signal_input_size=window_size,
            metmast_input_size=window_size,
            num_signal_feature=SIGNAL_SIZE,
            num_metmast_feature=METMAST_SIZE,
            output_size=OUTPUT_SIZE,
            dropout=DROPOUT,
        ).to(device)
    elif model_name == 'SFM':
        model = DualInputSFMModel(
            signal_dim=SIGNAL_SIZE,
            metmast_dim=METMAST_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_dim=OUTPUT_SIZE,
            dropout_W=DROPOUT,
            dropout_U=DROPOUT,
        ).to(device)
    elif model_name == 'GATS':
        model = DualInputGATModel(
            signal_dim=SIGNAL_SIZE,
            metmast_dim=METMAST_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            dropout=DROPOUT,
        ).to(device)
    else:
        raise ValueError(f'The parameter model_name should be LSTM/ALSTM/TCN/SFM/GATS, get {model_name} instead')

    # Define the optimizer and loss function based on specified parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if loss_name == 'BCE':
        criterion = nn.BCELoss()
    elif loss_name == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Parameter loss_name should be 'BCE'/'MSE', got '{}' instead".format(loss_name))

    # Train the model with the specified configurations.
    trained_model = train_model(model, train_signal_dataset, train_metmast_dataset, val_signal_dataset, val_metmast_dataset, \
                                optimizer, criterion, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device, clip_value=CLIP_VALUE, patience=PATIENCE)