"""
Data Loader Module
------------------
Responsible for loading dataset files and providing initial schema validation and inspection.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


class DataLoader:
    """Utility class to safely load and inspect raw datasets."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.raw_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads data from the given CSV path into a pandas DataFrame."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path.resolve()}")

        print(f"[DataLoader] Loading raw dataset from: {self.file_path.resolve()}")
        self.raw_df = pd.read_csv(self.file_path)
        print(f"[DataLoader] Successfully loaded {self.raw_df.shape[0]} rows and {self.raw_df.shape[1]} columns.")
        return self.raw_df

    def inspect_schema(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Inspects and returns metadata including shape, column types, and memory usage."""
        target_df = self.raw_df if df is None else df
        if target_df is None:
            raise ValueError("No dataframe loaded to inspect.")

        info_dict = {
            "rows": int(target_df.shape[0]),
            "columns": int(target_df.shape[1]),
            "memory_kb": round(target_df.memory_usage(deep=True).sum() / 1024, 2),
            "column_types": target_df.dtypes.astype(str).to_dict(),
            "columns_list": list(target_df.columns),
        }
        return info_dict
