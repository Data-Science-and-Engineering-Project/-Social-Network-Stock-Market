import pandas as pd
import numpy as np
from typing import Optional, List
from logger.logger import ETLLogger


class DataManipulation:
    """Handles data transformation, cleaning, and enrichment."""

    def __init__(self, logger: Optional[ETLLogger] = None):
        self.logger = logger or ETLLogger(name="DataManipulation")

    # ==================== MAIN ORCHESTRATION ====================

    def manipulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all manipulation operations in fixed sequence:
        1. Drop irrelevant columns
        2. Lowercase column names
        3. Clean data (strip cusip, remove duplicates, trim whitespace)
        4. Filter by period (2013_2Q and later)
        5. Add computed fields (value_per_share)
        6. Group and aggregate by cusip and period
        7. Handle outliers using IQR method
        8. Sort by value
        """

        self.logger.info("MANIPULATION PIPELINE - EXECUTING ALL STEPS")

        self.logger.info("[1/8] Converting column names to lowercase")
        df = self.lowercase_columns(df)

        self.logger.info("[1.5/8] Converting column names to remove underscore")
        df = self.remove_underscore(df)

        self.logger.info("[2/8] Dropping irrelevant columns")
        df = self.drop_irrelevant_columns(df)

        self.logger.info("[3/8] Cleaning data")
        df = self.clean_data(df)

        self.logger.info("[4/8] Filtering by period")
        df = self.filter_by_period(df)

        self.logger.info("[5/8] Adding computed fields")
        df = self.add_computed_fields(df)

        df = self.change_period_of_report_format(df)

        df = self.fix_column_typing_issue_with_median(df)

        df = df.drop(columns=['is_complete'])
        self.logger.info(f"MANIPULATION COMPLETE: {len(df)} records")
        return df

    # ==================== COLUMN OPERATIONS ====================

    def drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove irrelevant columns."""
        irrelevant_cols = [
            "titleofclass",
            "figi",
            "sshprnamttype",
            "investmentdiscretion",
            "othermanager",
            "votingauthoritysole",
            "votingauthorityshared",
            "votingauthoritynone",
            "votingauthsole",
            "votingauthshared",
            "votingauthnone",
            "submissiontype"
        ]
        cols_to_drop = [col for col in irrelevant_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        self.logger.info(f"Dropped {len(cols_to_drop)} irrelevant columns")
        return df

    def lowercase_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all column names to lowercase."""
        df.columns = df.columns.str.lower()
        self.logger.info(f"Lowercase: {df.shape[1]} columns standardized")
        return df

    def remove_underscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove underscores from all column names."""
        df.columns = df.columns.str.replace("_", "")
        self.logger.info(f"Underscores removed: {df.shape[1]} columns standardized")
        return df

    # ==================== DATA CLEANING ====================

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: strip cusip, remove nulls, duplicates, trim whitespace.

        Args:
            df: Input DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        self.logger.info("Cleaning data...")


        if "putcall" in df.columns:
            original_count = len(df)
            df = df[df["putcall"].isna() | (df["putcall"].str.strip() == "")]
            removed_putcall = original_count - len(df)
            if removed_putcall > 0:
                self.logger.info(f"Removed {removed_putcall} rows with putcall values")

            # Drop the put_call column
            df = df.drop(columns=["putcall"])
            self.logger.info("Dropped put_call column")

        # Strip whitespace from cusip
        if "cusip" in df.columns:
            df["cusip"] = df["cusip"].str.strip()

        # Remove duplicate rows
        original_count = len(df)
        df = df.drop_duplicates()
        removed_dupes = original_count - len(df)

        if removed_dupes > 0:
            self.logger.info(f"Removed {removed_dupes} duplicate rows")

        key_fields = ["cusip", "value"]
        for field in key_fields:
            if field in df.columns:
                df = df[df[field].notna()]

        self.logger.info(f"Clean: {len(df)} records remaining")
        return df

    # ==================== PERIOD FILTERING ====================
    def filter_by_period(
        self, df: pd.DataFrame, min_period: str = "2013_2Q"
    ) -> pd.DataFrame:
        """Filter records by period threshold (e.g., 2013_2Q and later)."""
        if "periodofreport" in df.columns:
            # Convert date string (e.g., "31-DEC-2013") to quarter format (e.g., "2013_Q4")
            df["periodofreport"] = (
                pd.to_datetime(df["periodofreport"], format="%d-%b-%Y").dt.year.astype(str)
                + "_Q"
                + ((pd.to_datetime(df["periodofreport"], format="%d-%b-%Y").dt.month - 1) // 3 + 1).astype(
                    str
                )
            )

            original_count = len(df)
            df = df[df["periodofreport"] >= min_period]
            filtered_count = original_count - len(df)
            self.logger.info(
                f"Filtered: removed {filtered_count} records before {min_period}"
            )
        return df

    # ==================== STANDARDIZATION ====================

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types.e."""
        self.logger.info("Standardizing columns...")

        # Convert value columns to numeric
        numeric_cols = [
            "value",
            "shares",
            "voting_sole",
            "voting_shared",
            "voting_none",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardize string columns to uppercase where appropriate
        string_cols = ["cusip", "share_type", "put_call", "discretion"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].str.upper()

        self.logger.info(f"Standardized: {df.shape[1]} columns")
        return df

    # ==================== COMPUTED FIELDS ====================

    def add_computed_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add computed/derived fields including value_per_share.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with computed fields.
        """
        self.logger.info("Adding computed fields...")

        # Add value_per_share
        if "value" in df.columns and "sshprnamt" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["sshprnamt"] = pd.to_numeric(df["sshprnamt"], errors="coerce")
            df["value_per_share"] = df["value"].astype(float) / df["sshprnamt"].astype(
                float
            )
            self.logger.info("Added value_per_share field")

        # Add data quality flag
        df["is_complete"] = (df.notna().sum(axis=1) >= df.shape[1] * 0.8).astype(int)

        self.logger.info(f"Added computed fields")
        return df


    def change_period_of_report_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year and quarter from periodofreport column (format: YYYY_QX)."""
        try:
            if "periodofreport" not in df.columns:
                self.logger.warning("periodofreport column not found")
                return df

            mask_valid = df["periodofreport"].notna() & (df["periodofreport"].str.len() > 0)

            df["year"] = pd.NA
            df["quarter"] = pd.NA

            valid_periods = df.loc[mask_valid, "periodofreport"].str.split("_", expand=True)

            if valid_periods.shape[1] >= 2:
                df.loc[mask_valid, "year"] = valid_periods[0].astype(int)
                df.loc[mask_valid, "quarter"] = valid_periods[1].str[1].astype(int)
                self.logger.info(f"Extracted year/quarter for {mask_valid.sum()} records")
            else:
                raise ValueError("periodofreport format invalid (expected YYYY_QX)")

            invalid_count = (~mask_valid).sum()
            if invalid_count > 0:
                self.logger.warning(f"Skipped {invalid_count} records with null/empty periodofreport")

            df = df.drop(columns=["periodofreport"])
            return df

        except ValueError as e:
            raise ValueError(f"Error in change_period_of_report_format: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in change_period_of_report_format: {str(e)}") from e

    def fix_column_typing_issue_with_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix value and value_per_share by using median value_per_share per year/quarter."""

        try:
            required_cols = ["year", "quarter", "value_per_share", "sshprnamt"]
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Missing required columns for median fix: {required_cols}")
                return df
            
            if df.empty:
                return df

            median_per_group = df.groupby(["year", "quarter"])["value_per_share"].transform("median")

            df["value_per_share"] = median_per_group

            df["value"] = df["value_per_share"] * df["sshprnamt"]
            
            self.logger.info(
                f"Fixed column typing: Applied median value_per_share for "
                f"{df.groupby(['year', 'quarter']).ngroups} (year, quarter) groups")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in fix_column_typing_issue_with_median: {str(e)}")
            raise

    # ==================== GROUPING & AGGREGATION ====================

    # def group_and_aggregate(
    #     self, df: pd.DataFrame, groupby_cols: Optional[List[str]] = None
    # ) -> pd.DataFrame:
    #     """
    #     Group by cusip and period, apply outlier handling per group, then recombine.
    #
    #     Args:
    #         df: Input DataFrame.
    #         groupby_cols: Columns to group by. Defaults to ['cusip', 'periodofreport'].
    #
    #     Returns:
    #         Aggregated DataFrame with outliers handled per group.
    #     """
    #     if groupby_cols is None:
    #         groupby_cols = ["cusip", "periodofreport"]
    #
    #     available_cols = [col for col in groupby_cols if col in df.columns]
    #
    #     if len(available_cols) > 0:
    #         self.logger.info(f"Grouping by: {available_cols}")
    #         df = df[df['cusip'] == '037833100']
    #         grouped = df.groupby(available_cols, as_index=False)
    #         processed_groups = []
    #
    #         for group_name, group_df in grouped:
    #             # Apply outlier handling to each group
    #             group_df = self.handle_outliers_iqr(group_df, column="value_per_share")
    #             processed_groups.append(group_df)
    #
    #         # Recombine all groups into single DataFrame
    #         df = pd.concat(processed_groups, ignore_index=True)
    #         self.logger.info(
    #             f"Aggregated to {len(grouped)} groups with outlier handling applied"
    #         )
    #
    #     return df

    # ==================== OUTLIER HANDLING ====================

    # def handle_outliers_iqr(
    #     self, df: pd.DataFrame, column: str = "value_per_share"
    # ) -> pd.DataFrame:
    #     """
    #     Handle outliers using IQR method (robust to extreme values).
    #     Adjusts outliers by multiplying with powers of 10 to bring values closer to the average.
    #
    #     Args:
    #         df: Input DataFrame.
    #         column: Column to detect outliers in.
    #
    #     Returns:
    #         DataFrame with outliers handled.
    #     """
    #     if column not in df.columns:
    #         self.logger.warning(f"Column '{column}' not found")
    #         return df
    #
    #     # Remove NaN and infinite values
    #     mask_valid = df[column].notna() & np.isfinite(df[column])
    #
    #     Q1 = df.loc[mask_valid, column].quantile(0.1)
    #     Q3 = df.loc[mask_valid, column].quantile(0.9)
    #     IQR = Q3 - Q1
    #
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #
    #     # Adjust outliers by multiplying with powers of 10
    #     iteration = df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column].items()
    #     for index, value in iteration:
    #         if value < lower_bound:
    #             multiplier = 10 ** (1)  # Example for positive adjustment
    #         else:
    #             multiplier = 10 ** (-1)  # Example for negative adjustment
    #         df.at[index, column] = value * multiplier
    #
    #     self.logger.info(
    #         f"Outliers (IQR): adjusted values to bounds "
    #         f"([{lower_bound:.4f}, {upper_bound:.4f}])"
    #     )
    #     return df
    #
    # def handle_outliers_zscore(
    #     self, df: pd.DataFrame, column: str = "value_per_share", threshold: float = 3.0
    # ) -> pd.DataFrame:
    #     """
    #     Alternative method: Handle outliers using Z-score.
    #     Removes values where |z-score| > threshold (default 3 = ~0.3% outliers).
    #
    #     Args:
    #         df: Input DataFrame.
    #         column: Column to detect outliers in.
    #         threshold: Z-score threshold (default 3).
    #
    #     Returns:
    #         DataFrame with outliers handled.
    #     """
    #     if column not in df.columns:
    #         self.logger.warning(f"Column '{column}' not found")
    #         return df
    #
    #     mask_valid = df[column].notna() & np.isfinite(df[column])
    #     mean = df.loc[mask_valid, column].mean()
    #     std = df.loc[mask_valid, column].std()
    #
    #     if std == 0:
    #         self.logger.warning(f"Zero std deviation for column '{column}'")
    #         return df
    #
    #     z_scores = np.abs((df[column] - mean) / std)
    #     original_count = len(df)
    #     df = df[z_scores <= threshold]
    #     removed_count = original_count - len(df)
    #
    #     self.logger.info(f"Outliers (Z-score): removed {removed_count} records")
    #     return df
    #
    # # ==================== FILTERING ====================
    #
    # def filter_by_value(self, df: pd.DataFrame, min_value: float = 0) -> pd.DataFrame:
    #     """Filter holdings by minimum value threshold."""
    #     if "value" in df.columns:
    #         original_count = len(df)
    #         df = df[df["value"] >= min_value]
    #         filtered_count = original_count - len(df)
    #         self.logger.info(
    #             f"Filtered: removed {filtered_count} holdings below {min_value}"
    #         )
    #     return df
    #
    # # ==================== SORTING ====================
    #
    # def sort_by_value(self, df: pd.DataFrame, descending: bool = True) -> pd.DataFrame:
    #     """Sort holdings by value."""
    #     if "value" in df.columns:
    #         df = df.sort_values("value", ascending=not descending)
    #         self.logger.info(f"Sorted by value ({('DESC' if descending else 'ASC')})")
    #     return df
