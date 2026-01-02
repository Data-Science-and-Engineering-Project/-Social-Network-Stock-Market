import pandas as pd
import os
from ETL.load.postgres_loader import PostgresLoader
from logger.logger import ETLLogger


class DataLoader:

    def __init__(self, output_dir: str = "13f_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.postgres = PostgresLoader()

    def load_to_db(self, df: pd.DataFrame) -> None:
        """
        Load DataFrame to PostgreSQL holding table.

        Args:
            df: DataFrame to load
        """
        self.postgres.load(df, table_name="holdings", if_exists="append")