import json
from typing import Dict
from logger.logger import ETLLogger
from datetime import datetime


class ETLUtils:

    @staticmethod
    def load_and_flatten_nested_dict(
        json_path: str, logger: ETLLogger, separator: str = "_"
    ) -> Dict[str, str]:
        """
        Load JSON file and flatten nested structure.

        Example: {"2025": {"Q2": "file.zip"}} → {"2025_Q2": "file.zip"}
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            flat_dict = {}
            for key1, nested_dict in data.items():
                for key2, value in nested_dict.items():
                    flat_key = f"{key1}{separator}{key2}"
                    flat_dict[flat_key] = value

            logger.info(f"Loaded {len(flat_dict)} entries from {json_path}")
            return flat_dict
        except FileNotFoundError:
            logger.error(f"File not found: {json_path}")
            raise FileNotFoundError(f"File not found: {json_path}")
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise ValueError(f"Error loading JSON file: {e}")

    @staticmethod
    def quarter_string_to_date(quarter: str) -> datetime:
        """
        Convert quarter string (e.g., '2025_Q1') to datetime (e.g., 2025-03-31).
        """
        quarter = quarter[0]
        parts = quarter.split("_")
        year = int(parts[0])
        q = int(parts[1][1])  # Extract number from 'Q1', 'Q2', etc.

        # Quarter end months: Q1->3, Q2->6, Q3->9, Q4->12
        month = q * 3
        day = 31  # Simplified; adjust if needed for exact quarter ends

        return datetime(year, month, day)
