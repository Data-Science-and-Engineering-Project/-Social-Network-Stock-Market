from abc import ABC, abstractmethod
import pandas as pd


class ExtractionStrategy(ABC):
    """
    Base class for ALL extraction strategies.
    Each extractor returns a DataFrame of extracted data.
    """

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        pass
