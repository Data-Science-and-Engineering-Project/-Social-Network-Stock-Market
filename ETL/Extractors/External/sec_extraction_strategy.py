import requests
import zipfile
import glob
import os
import tempfile
import shutil
import re
import json
from typing import List, Dict, Any, Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from Extractors.base_strategy import ExtractionStrategy
from dal.dal import DAL
from data_handlers.web_data_fetcher import RemoteFileFetcher
from data_handlers.db_data_handler.db_abstract import AbstractDBHandler
from logger.logger import ETLLogger
from ETL.utils.utils import ETLUtils


class SECExtractionStrategy(ExtractionStrategy):
    """
    Extracts 13F filings from SEC quarterly dataset ZIPs.
    Loads quarterly dataset mappings from JSON configuration.
    """

    BASE_URL = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
    INFOTABLE_PATTERN = re.compile(r"(?i)^infotable\.tsv$")
    SUBMISSION_PATTERN = re.compile(r"(?i)^submission\.tsv$")

    def __init__(
        self,
        quarters: Optional[List[str]] = None,
        output_dir: str = "13f_outputs",
        cik_filter: Optional[str] = None,
        config_path: str = "data/quarterly_datasets.json",
        dal: Optional[DAL] = None,
        logger: Optional[ETLLogger] = None,
    ):
        self.data_lock = Lock()
        self.output_dir = output_dir
        self.cik_filter = cik_filter
        self.file_fetcher = RemoteFileFetcher()
        self.logger = logger or ETLLogger(name="SECExtractionStrategy")
        os.makedirs(self.output_dir, exist_ok=True)

        # load quarterly datasets from JSON
        self.quarterly_datasets = ETLUtils.load_and_flatten_nested_dict(config_path, self.logger)

        self.quarters = quarters or self._get_all_quarters()

    def _get_all_quarters(self) -> List[str]:
        """Get all available quarters from loaded config."""
        return sorted(self.quarterly_datasets.keys())

    # ==================== MAIN EXTRACTION ====================

    def extract(self) -> pd.DataFrame:
        """Main extraction orchestration."""
        temp_dir = tempfile.mkdtemp(prefix="sec_13f_quarterly_")
        all_quarters_data = []

        try:
            ETLLogger().info(f"Extracting {len(self.quarters)} quarters...")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self.extract_helper, quarter, temp_dir, all_quarters_data
                    ): quarter
                    for quarter in self.quarters
                }

                for future in as_completed(futures):
                    quarter = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        ETLLogger().error(f"Thread exception for {quarter}: {e}")

            # Combine all quarters
            if not all_quarters_data:
                ETLLogger().error("No quarters processed successfully")
                raise ValueError("No quarters processed successfully")

            combined_df = pd.concat(all_quarters_data, ignore_index=True)

            ETLLogger().info(f"Total holdings extracted: {len(combined_df)}")

            return combined_df

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                ETLLogger().info("Cleaned up temp directory")

    def extract_helper(
        self, quarter: str, temp_dir: str, all_quarters_data: List
    ) -> None:
        """Thread worker: process single quarter and append to results with lock."""
        if quarter not in self.quarterly_datasets:
            ETLLogger().warning(f"Unknown quarter: {quarter}, skipping...")
            return

        try:
            quarter_df = self._process_quarter(quarter, temp_dir)
            with self.data_lock:
                all_quarters_data.append(quarter_df)
        except Exception as e:
            ETLLogger().error(f"Failed to process {quarter}: {e}")
            ETLLogger().exception(f"Exception details for {quarter}:")

    # ==================== DOWNLOAD FUNCTIONS ====================

    def _download_zip(self, url: str, output_path: str) -> None:
        """Download SEC quarterly ZIP file with streaming and progress tracking via dal."""
        ETLLogger().info(f"Downloading from: {url}")

        try:
            response = self.file_fetcher.fetch_stream(url)

            def on_progress(written: int, total: int):
                if total:
                    pct = (written / total) * 100
                    ETLLogger().debug(f"{pct:.1f}% downloaded...")

            with open(output_path, "wb") as f:
                self.file_fetcher.write_chunks_to_file(response, f, on_progress)

            ETLLogger().info(f"Downloaded: {output_path}")
        except Exception as e:
            ETLLogger().error(f"Download failed: {e}")
            ETLLogger().exception("Download error details:")
            raise

    def _ensure_zip_downloaded(self, quarter: str) -> str:
        """Ensure ZIP file exists; download if missing. Returns path to ZIP."""
        zip_filename = self.quarterly_datasets[quarter]
        zip_url = self.BASE_URL + zip_filename
        zip_path = os.path.join(self.output_dir, zip_filename)

        if not os.path.exists(zip_path):
            self._download_zip(zip_url, zip_path)
        else:
            ETLLogger().info(f"Already downloaded: {zip_filename}")

        return zip_path

    # ==================== EXTRACTION FUNCTIONS ====================

    def _extract_zip(self, zip_path: str, extract_to: str) -> None:
        """Extract ZIP file to target directory."""
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        ETLLogger().info(f"Extracted to: {extract_to}")

    # ==================== TSV PARSING FUNCTIONS ====================

    def _rename_tsv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename TSV columns to standardized names."""
        column_map = {
            "CUSIP": "cusip",
            "Name of Issuer": "nameOfIssuer",
            "Market Value (x$1000)": "value",
            "Shrs or Prin Amt": "shares",
            "Sh/Prn": "share_type",
            "Inv. Discretion": "investment_discretion",
            "Put/Call": "put_call",
            "Sole Voting": "voting_sole",
            "Shared Voting": "voting_shared",
            "No Voting": "voting_none",
        }
        return df.rename(columns=column_map)

    def _read_tsv_file(self, tsv_file: str) -> Optional[pd.DataFrame]:
        """Read single TSV file using pandas (fast C engine)."""
        try:
            df = pd.read_csv(
                tsv_file,
                sep="\t",
                dtype=str,
                na_filter=False,
                engine="c",
            )
            return df
        except Exception as e:
            ETLLogger().error(f"Error reading {tsv_file}: {e}")
            return None

    def _parse_tsv_file(self, tsv_file: str) -> Optional[pd.DataFrame]:
        """Parse TSV file and rename columns."""
        df = self._read_tsv_file(tsv_file)
        if df is None or df.empty:
            return None

        df = self._rename_tsv_columns(df)
        return df

    # ==================== FOLDER OPERATIONS ====================

    def _find_tsv_files(self, folder: str, pattern: re.Pattern) -> List[str]:
        """Find TSV files matching pattern in folder recursively."""
        all_files = glob.glob(f"{folder}/**/*.tsv", recursive=True)
        matching_files = [f for f in all_files if pattern.match(os.path.basename(f))]
        return matching_files

    def _read_specific_tsv_files(
        self, folder: str, pattern: re.Pattern
    ) -> List[pd.DataFrame]:
        """Read all TSV files matching pattern from folder."""
        tsv_files = self._find_tsv_files(folder, pattern)
        ETLLogger().info(f"Found {len(tsv_files)} TSV files")

        dataframes = []
        for i, tsv_file in enumerate(tsv_files, 1):
            if i % 10 == 0:
                ETLLogger().debug(f"Parsed {i}/{len(tsv_files)} files...")

            df = self._parse_tsv_file(tsv_file)
            if df is not None:
                dataframes.append(df)

        return dataframes

    # ==================== MERGE OPERATIONS ====================

    def _merge_infotable_and_submission(
        self, info_dfs: List[pd.DataFrame], submission_dfs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge infotable with submission data on ACCESSION_NUMBER."""
        if not info_dfs or not submission_dfs:
            ETLLogger().error("Missing infotable or submission data")
            raise ValueError("Missing infotable or submission data")

        infotable = pd.concat(info_dfs, ignore_index=True)
        submission = pd.concat(submission_dfs, ignore_index=True)

        merged = pd.merge(infotable, submission, how="inner", on="ACCESSION_NUMBER")

        return merged

    def _apply_cik_filter(self, df: pd.DataFrame, cik: str) -> pd.DataFrame:
        """Filter dataframe by CIK if provided."""
        if cik and "CIK" in df.columns:
            original_count = len(df)
            df = df[df["CIK"] == cik]
            filtered_count = len(df)
            ETLLogger().info(
                f"CIK filter: {original_count} → {filtered_count} rows (CIK: {cik})"
            )
        return df

    def _apply_cusip_filter(self, df: pd.DataFrame, cusip: str) -> pd.DataFrame:
        """Filter dataframe by CUSIP if provided."""
        if cusip and "cusip" in df.columns:
            original_count = len(df)
            df["cusip"] = df["cusip"].str.strip()
            df = df[df["cusip"] == cusip]
            filtered_count = len(df)
            ETLLogger().info(
                f"CUSIP filter: {original_count} → {filtered_count} rows (CUSIP: {cusip})"
            )
        return df
    # ==================== QUARTER PROCESSING ====================

    def _process_quarter(self, quarter: str, temp_dir: str) -> pd.DataFrame:
        """Process single quarter: download, extract, merge, filter."""
        ETLLogger().info(f"Processing {quarter}...")

        # Download if needed
        zip_path = self._ensure_zip_downloaded(quarter)

        # Extract
        extract_dir = os.path.join(temp_dir, quarter)
        self._extract_zip(zip_path, extract_dir)

        # Parse infotable
        ETLLogger().info("Parsing infotable files...")
        info_dfs = self._read_specific_tsv_files(extract_dir, self.INFOTABLE_PATTERN)

        # Parse submission
        ETLLogger().info("Parsing submission files...")
        submission_dfs = self._read_specific_tsv_files(
            extract_dir, self.SUBMISSION_PATTERN
        )

        # Merge
        merged_df = self._merge_infotable_and_submission(info_dfs, submission_dfs)

        # Apply CIK filter if provided
        if self.cik_filter:
            merged_df = self._apply_cik_filter(merged_df, self.cik_filter)

        return merged_df
