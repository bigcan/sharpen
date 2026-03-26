import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TimeRange:
    start: str
    end: str

    def to_tuple(self):
        return (self.start, self.end)

class RollingWindowSplitter:
    """
    Implements the 3-Split (Train/Validation/Trade) Rolling Window strategy
    for non-stationary financial time-series data.
    """
    def __init__(
        self,
        train_months: int = 12,
        val_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        buffer_days: int = 0,
    ):
        """
        Args:
            train_months: Length of training window.
            val_months: Length of validation window (immediately follows train).
            test_months: Length of testing/trade window (immediately follows val).
            step_months: How much to shift the window forward for the next fold.
            buffer_days: Gap between windows (optional, usually 0).
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        self.buffer_days = buffer_days

    def split(self, start_date: str, end_date: str) -> list[dict[str, TimeRange]]:
        """
        Generates rolling window splits within the global start/end range.

        Returns:
            List of dicts: [{'train': (s,e), 'val': (s,e), 'test': (s,e)}, ...]
        """
        folds = []
        current_start = pd.Timestamp(start_date)
        global_end = pd.Timestamp(end_date)

        while True:
            # 1. Calculate boundaries
            train_end = current_start + pd.DateOffset(months=self.train_months)

            val_start = train_end + pd.Timedelta(days=self.buffer_days)
            val_end = val_start + pd.DateOffset(months=self.val_months)

            test_start = val_end + pd.Timedelta(days=self.buffer_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # 2. Check if we exceeded global end
            if test_end > global_end:
                break

            # 3. Add Fold
            folds.append({
                "train": TimeRange(str(current_start), str(train_end)),
                "val": TimeRange(str(val_start), str(val_end)),
                "test": TimeRange(str(test_start), str(test_end)),
            })

            # 4. Step forward
            current_start += pd.DateOffset(months=self.step_months)

        return folds

    @staticmethod
    def print_schedule(folds):
        logger.info(f"{'Fold':<5} | {'Train':<25} | {'Validation':<25} | {'Trade (Test)':<25}")
        logger.info("-" * 85)
        for i, fold in enumerate(folds):
            t = fold['train']
            v = fold['val']
            e = fold['test']
            logger.info(f"{i+1:<5} | {t.start[:10]} -> {t.end[:10]} | {v.start[:10]} -> {v.end[:10]} | {e.start[:10]} -> {e.end[:10]}")

# Example Usage
if __name__ == "__main__":
    splitter = RollingWindowSplitter(train_months=3, val_months=1, test_months=1, step_months=1)
    folds = splitter.split("2023-01-01", "2023-12-31")
    RollingWindowSplitter.print_schedule(folds)
