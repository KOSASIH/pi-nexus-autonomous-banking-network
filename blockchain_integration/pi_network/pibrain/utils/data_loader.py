# data_loader.py

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from typing import Any, Dict, List, Optional

class DataLoader:
    """Data loader class."""

    def __init__(self, data_dir: str, file_pattern: str, chunk_size: int = 1000):
        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.chunk_size = chunk_size
        self.file_list = self._get_file_list()

    def _get_file_list(self) -> List[str]:
        """Get a list of files matching the file pattern."""
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.startswith(self.file_pattern)]

    def load_data(self) -> pd.DataFrame:
        """Load data from files."""
        data = []
        for file in self.file_list:
            chunk = pd.read_csv(file, chunksize=self.chunk_size)
            data.extend(chunk)
        data = pd.concat(data, ignore_index=True)
        return data

    def load_data_in_chunks(self) -> Iterator[pd.DataFrame]:
        """Load data in chunks."""
        for file in self.file_list:
            chunk = pd.read_csv(file, chunksize=self.chunk_size)
            yield chunk

    def shuffle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle the data."""
        return shuffle(data)

def main():
    data_dir = 'data'
    file_pattern = 'data_'
    loader = DataLoader(data_dir, file_pattern)
    data = loader.load_data()
    print(data.head())

if __name__ == '__main__':
    main()
