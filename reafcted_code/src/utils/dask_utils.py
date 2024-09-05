# utils/dask_utils.py
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import gc

class DaskManager:
    def __init__(self, cfg):
        """Initialize DaskManager with configurations from Hydra."""
        self.dynamic_partitioning = cfg.dask.dynamic_partitioning
        self.npartitions = cfg.dask.npartitions
        self.max_tasks = cfg.dask.max_tasks
        self.client = None

    def initialize_client(self):
        """Initialize Dask client with dynamic partitioning based on config."""
        self.client = Client(n_workers=self.npartitions, threads_per_worker=1)
        return self.client

    def close_client(self):
        """Close Dask client and perform memory cleanup."""
        if self.client:
            self.client.close()
        gc.collect()

    def pandas_to_dask(self, df, rows_per_partition=100000):
        """Convert a Pandas DataFrame to a Dask DataFrame with dynamic partitioning."""
        if self.dynamic_partitioning:
            npartitions = max(1, min(self.max_tasks, len(df) // rows_per_partition))
        else:
            npartitions = self.npartitions
        return dd.from_pandas(df, npartitions=npartitions)

    def run_with_progress(self, dask_obj):
        """Run a Dask operation with a progress bar."""
        with ProgressBar():
            result = dask_obj.compute()
        return result
