import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

################################################################
class KinematicsDataset(Dataset):
    def __init__(self, dataframe, leave_out_subject, feature_cols, target_col, subject_col):
        """
        Custom PyTorch Dataset for leave-one-subject-out cross-validation.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame with features, subject column, and target column.
            leave_out_subject (str): Subject to leave out for validation/testing.
            feature_cols (list): List of feature column names.
            target_col (str): Name of the target column (pain level, protected behavior, or both).
            subject_col (str): Name of the subject column.
        """
        # Split data into training and test sets
        self.train_data = dataframe[dataframe[subject_col] != leave_out_subject]
        self.test_data = dataframe[dataframe[subject_col] == leave_out_subject]

        # Extract features and targets
        self.features = self.train_data[feature_cols].values
        self.targets = self.train_data[target_col].values
        self.test_features = self.test_data[feature_cols].values
        self.test_targets = self.test_data[target_col].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a single sample from the training dataset.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (feature_tensor, target_tensor)
        """
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target

    def get_test_data(self):
        """
        Get the test set for the current subject.
        
        Returns:
            tuple: (test_features_tensor, test_targets_tensor)
        """
        test_features = torch.tensor(self.test_features, dtype=torch.float32)
        test_targets = torch.tensor(self.test_targets, dtype=torch.float32)
        return test_features, test_targets

class _trivial_scaler():
    def __init__(self,):
        return
    
    def fit(self, x):
        return self
    
    def transform(self, x):
        return x

# Example usage
def prepare_dataset(dataframe, feature_cols, target_col, subject_col):
    """
    Generate leave-one-subject-out splits and return datasets for each subject.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column (pain level).
        subject_col (str): Name of the subject column.
    
    Returns:
        dict: A dictionary with subjects as keys and corresponding datasets as values.
    """
    datasets = {}
    subjects = dataframe[subject_col].unique()
    
    for subject in subjects:
        datasets[subject] = KinematicsDataset(dataframe, subject, feature_cols, target_col, subject_col)
    
    return datasets


class SlidingWindowDataset(Dataset):
    def __init__(self, dataframe, leave_out_subject, feature_cols, target_col, subject_col, window_length, step_size, 
                targe_scaling=False, sequence_majority_voting=True):
        """
        Custom PyTorch Dataset with sliding window for leave-one-subject-out cross-validation.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with features, subject column, and target column.
            leave_out_subject (str): Subject to leave out for validation/testing.
            feature_cols (list): List of feature column names.
            target_col (str): Name of the target column (pain level).
            subject_col (str): Name of the subject column.
            window_length (int): Length of each sliding window.
            step_size (int): Step size for sliding window.
        """
        scaler = StandardScaler
        self.window_length = window_length
        self.step_size = step_size
        self.feature_num = len(feature_cols)

        # Split data into training and test sets
        self.train_data = dataframe[dataframe[subject_col] != leave_out_subject]
        self.test_data = dataframe[dataframe[subject_col] == leave_out_subject]

        # Generate sliding windows
        self.train_windows = self._generate_windows(self.train_data, feature_cols, target_col, subject_col)
        self.test_windows = self._generate_windows(self.test_data, feature_cols, target_col, subject_col)
        # scale data
        train_features, train_targets = self.train_windows
        test_features, test_targets = self.test_windows
        feature_scaler = scaler().fit(train_features.reshape(-1, self.feature_num))
        target_scaler = scaler().fit(train_targets.squeeze()) if targe_scaling else _trivial_scaler()
        self.train_windows = feature_scaler.transform(train_features.reshape(-1, self.feature_num)).reshape(*train_features.shape), target_scaler.transform(train_targets.squeeze()).reshape(*train_targets.shape)
        self.test_windows = feature_scaler.transform(test_features.reshape(-1, self.feature_num)).reshape(*test_features.shape), target_scaler.transform(test_targets.squeeze()).reshape(*test_targets.shape)
        self.feature_scaler, self.target_scaler = feature_scaler, target_scaler # save scaler

    def _generate_windows(self, data, feature_cols, target_col, subject_col):
        """
        Generate sliding window samples using numpy stride tricks.

        Args:
            data (pd.DataFrame): Data subset for training or testing.
            feature_cols (list): List of feature column names.
            target_col (str): Target column name.
            subject_col (str): Subject column name.

        Returns:
            tuple: Features and targets for all windows.
        """
        features_list = []
        targets_list = []

        subjects = data[subject_col].unique()

        for subject in subjects:
            subject_data = data[data[subject_col] == subject]
            features = subject_data[feature_cols].values
            targets = subject_data[target_col].values
            features = features.reshape(*features.shape[:1], -1)
            targets = targets.reshape(*targets.shape[:1], -1)  # make sure targets have the same shape as features
            if len(features) < self.window_length:
                    # TODO: this is a very rare case handling, not the implementation for zero padding
                # Handle case where data is shorter than window length
                padded_features = np.zeros((self.window_length, features.shape[1]), dtype=np.float32)
                padded_targets = np.zeros((self.window_length, targets.shape[1]), dtype=np.float32)
                padded_features[:len(features)] = features
                padded_targets[:len(targets)] = targets

                features_list.append(padded_features[np.newaxis, ...])  # Add new axis to match shape
                targets_list.append(padded_targets[np.newaxis, ...])  # Add new axis to match shape
            # Generate sliding windows
            feature_windows = np.lib.stride_tricks.sliding_window_view(
                features, (self.window_length, features.shape[-1])
            )[::self.step_size, 0, :, :]
            target_windows = np.lib.stride_tricks.sliding_window_view(
                targets, (self.window_length, targets.shape[-1])
            )[::self.step_size, 0, :, :]

            features_list.append(feature_windows)
            targets_list.append(target_windows)

        # Concatenate all windows
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        all_targets = all_targets.mean(axis=-2)
        return all_features, all_targets    # shape: (N, L, D) where L is the sequence length and D the number of features

    def __len__(self):
        return len(self.train_windows[0])

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (feature_tensor, target_tensor)
        """
        feature_window = self.train_windows[0][idx]
        target_window = self.train_windows[1][idx]
        return (
            torch.tensor(feature_window, dtype=torch.float32),
            torch.tensor(target_window, dtype=torch.float32)
        )

    def get_test_data(self):
        """
        Get the test set for the current subject.

        Returns:
            tuple: Test features and targets.
        """
        test_features, test_targets = self.test_windows
        return (
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_targets, dtype=torch.float32)
        )
    
    def get_train_data(self):
        """
        Get the test set for the current subject.

        Returns:
            tuple: Test features and targets.
        """
        train_features, train_targets = self.train_windows
        return (
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32)
        )

# Example usage
def prepare_sliding_window_dataset(dataframe, feature_cols, target_col, subject_col, window_length, step_size):
    """
    Generate leave-one-subject-out sliding window datasets for each subject.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column.
        subject_col (str): Name of the subject column.
        window_length (int): Length of each sliding window.
        step_size (int): Step size for the sliding window.

    Returns:
        dict: Dictionary of datasets keyed by subject.
    """
    datasets = {}
    subjects = dataframe[subject_col].unique()
    
    for subject in subjects:
        datasets[subject] = SlidingWindowDataset(
            dataframe, subject, feature_cols, target_col, subject_col, window_length, step_size
        )
    
    return datasets


def create_weighted_sampler(dataset, num_bins=10):
    """
    Create a weighted sampler for imbalanced pain levels in the dataset.

    Args:
        dataset (Dataset): The PyTorch dataset (e.g., SlidingWindowDataset).
        num_bins (int): Number of bins to discretize the pain levels (default: 10).

    Returns:
        WeightedRandomSampler: A sampler that weights samples based on bin frequencies.
    """
    # Get all targets from the dataset
    all_targets = []
    for _, target in dataset:
        target = target.numpy().squeeze().mean(keepdims=True)  #
        if len(target.shape) <= 1:
            target = np.array([target])  # expand dimension
        all_targets.extend(target)
    all_targets = np.array(all_targets)

    # Discretize pain levels into bins
    bin_edges = np.linspace(0, 1, num_bins + 1)  # Create bin edges from 0 to 1
    bins = np.digitize(all_targets, bins=bin_edges, right=True)  # Assign each pain level to a bin

    # Calculate bin frequencies
    bin_counts = np.bincount(bins, minlength=num_bins + 1)  # Add +1 for edge cases
    bin_counts = bin_counts[1:]  # Ignore bin 0 (values < bin_edges[0])

    # Compute weights inversely proportional to bin frequencies
    bin_weights = 1.0 / (bin_counts + 1e-6)  # Add small epsilon to avoid division by zero
    sample_weights = bin_weights[bins - 1]  # Assign weight to each sample based on its bin

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),  # Total number of samples
        replacement=True  # Allow replacement for balanced sampling
    )

    return sampler