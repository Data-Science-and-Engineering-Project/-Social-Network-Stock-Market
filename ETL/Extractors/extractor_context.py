from typing import Any, Dict, Optional
import pandas as pd
from Extractors.base_strategy import ExtractionStrategy
from Extractors.External.sec_extraction_strategy import SECExtractionStrategy


class ExtractorContext:
    """
    Context that creates and executes extraction strategies.
    """

    STRATEGY_MAP = {
        "sec": SECExtractionStrategy,
        # "csv": CSVExtractionStrategy,
        # "xml": XMLExtractionStrategy,
    }

    def __init__(self, extractor_type: str, **kwargs):
        """
        Initialize context with strategy type and configuration.

        Args:
            extractor_type: Type of extractor (e.g., "sec", "csv", "xml")
            **kwargs: Configuration parameters for the strategy
        """
        self.strategy = self._create_strategy(extractor_type, **kwargs)

    def _create_strategy(self, extractor_type: str, **kwargs) -> ExtractionStrategy:
        """Factory method to create strategy by type."""
        if extractor_type not in self.STRATEGY_MAP:
            raise ValueError(
                f"Unknown extractor type: {extractor_type}. "
                f"Available: {list(self.STRATEGY_MAP.keys())}"
            )

        strategy_class = self.STRATEGY_MAP[extractor_type]
        return strategy_class(**kwargs)

    def set_strategy(self, strategy: ExtractionStrategy):
        self.strategy = strategy

    def execute(self) -> pd.DataFrame:
        """Execute the extraction strategy and return DataFrame."""
        return self.strategy.extract()
