"""Module: splitter
Purpose: Provide robust data splitting logic with purging and embargoing for time-series data."""

import pandas as pd
from typing import Tuple, List

class DataSplitter:
    """
    Splits time-series data into Train, Validation, and Test sets with purging and embargoing.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "date"):
        """
        Args:
            df: The dataframe containing the time-series data.
            date_col: The name of the column containing the date information.
        """
        self.df = df.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col

    def split_by_date(self, train_start: str, train_end: str, 
                      val_start: str, val_end: str, 
                      test_start: str, test_end: str,
                      purge_overlap: int = 0, embargo: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into Train, Validation, and Test sets based on date ranges.
        
        Args:
            train_start: Start date for training set (inclusive).
            train_end: End date for training set (inclusive).
            val_start: Start date for validation set (inclusive).
            val_end: End date for validation set (inclusive).
            test_start: Start date for test set (inclusive).
            test_end: End date for test set (inclusive).
            purge_overlap: Number of observations to drop from the end of the training set 
                           before the validation set starts (to prevent leakage).
            embargo: Number of observations to drop from the beginning of the test set 
                     after the training/validation set ends.

        Returns:
            A tuple of (train_df, val_df, test_df).
        """
        train_mask = (self.df[self.date_col] >= train_start) & (self.df[self.date_col] <= train_end)
        val_mask = (self.df[self.date_col] >= val_start) & (self.df[self.date_col] <= val_end)
        test_mask = (self.df[self.date_col] >= test_start) & (self.df[self.date_col] <= test_end)

        train_df = self.df.loc[train_mask].copy()
        val_df = self.df.loc[val_mask].copy()
        test_df = self.df.loc[test_mask].copy()

        # Apply purging (removing data from end of train set)
        if purge_overlap > 0:
             # Assuming sorted data, drop last 'purge_overlap' rows from train
             if len(train_df) > purge_overlap:
                 train_df = train_df.iloc[:-purge_overlap]
             else:
                 raise ValueError(f"Training set is smaller than purge_overlap ({purge_overlap})")

        # Apply embargoing (removing data from beginning of test set)
        # Embargo is typically applied after the training set to prevent leakage from the 
        # immediate future into the training set if there's overlap in labeling logic.
        # Here we apply it to the test set relative to the validation set (standard walk-forward).
        if embargo > 0:
            if len(test_df) > embargo:
                test_df = test_df.iloc[embargo:]
            else:
                 raise ValueError(f"Test set is smaller than embargo ({embargo})")

        return train_df, val_df, test_df

    def get_rolling_splits(self, start_date: str, end_date: str, 
                           train_window_months: int, val_window_months: int, test_window_months: int,
                           step_months: int) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generates rolling window splits.
        
        (Implementation pending - for Phase 4)
        """
        raise NotImplementedError("Rolling splits are not yet implemented.")
