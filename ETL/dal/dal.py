from data_handlers.web_data_fetcher import RemoteFileFetcher
from load.load import DataLoader


class DAL:
    """
    Data Access Layer responsible only for database operations.
    Wraps a DB handler that implements AbstractDBHandler.
    """
    db_handler = DataLoader()

    @staticmethod
    def load_data(df):
        """Load data into the database using the DataLoader"""
        DAL.db_handler.load_to_db(df)





