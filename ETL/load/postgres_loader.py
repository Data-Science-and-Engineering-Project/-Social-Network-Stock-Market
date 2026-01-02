import pandas as pd
from typing import Optional
from data_handlers.db_data_handler.postgres_handler import PostgresHandler
from logger.logger import ETLLogger


class PostgresLoader:
    """Handles loading DataFrames to PostgreSQL database using PostgresHandler."""

    def __init__(self):
        self.handler = PostgresHandler()

    def load(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "append"
    ) -> bool:
        """
        Load DataFrame to PostgreSQL table.

        Args:
            df: DataFrame to load
            table_name: Target table name
            if_exists: 'fail', 'replace', or 'append'

        Returns:
            True if successful, False otherwise
        """
        try:
            insert_count = self.handler.insert_dataframe(df, table_name)
            return insert_count > 0
        except Exception as e:
            ETLLogger().error(f"PostgreSQL load failed: {str(e)}")
            return False
