#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ETL.Extractors.extractor_context import ExtractorContext
from ETL.dal.dal import DAL
from manipulation.manipulation import DataManipulation
from load.load import DataLoader
from logger.logger import ETLLogger
from dotenv import load_dotenv
import json
from ETL.utils.utils import ETLUtils
from ETL.utils import *

load_dotenv()
debug_mode = False

def main():
    def load_quarters_from_json(config_path: str = "data/run.json") -> list:
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            quarters = config.get("all_years_and_quarters", [])
            if not quarters:
                ETLLogger().error(f"No quarters found in {config_path}")
                return []
            ETLLogger().info(f"Loaded {len(quarters)} quarters from {config_path}")
            return quarters
        except FileNotFoundError:
            ETLLogger().error(f"Config file not found: {config_path}")
            return []
        except json.JSONDecodeError:
            ETLLogger().error(f"Invalid JSON in: {config_path}")
            return []

    running_lst = load_quarters_from_json()
    #QA
    # if debug_mode:
    # running_lst = [running_lst[0]]
    for quarter in running_lst:
        etl(quarter)
    return 0

def etl(quarter):
    """Main ETL pipeline execution with integrated partition handling."""

    # ==================== INITIALIZATION ====================
    ETLLogger(name="ETL_Pipeline", console_output=True)

    # ==================== EXTRACT ====================
    ETLLogger().info("=" * 80)
    ETLLogger().info("STAGE 1: EXTRACTION")
    ETLLogger().info("=" * 80)

    try:
        context = ExtractorContext(extractor_type="sec", quarters=quarter)
        df = context.execute()

        ETLLogger().info(f"Extraction complete: {len(df)} records")
    except Exception as e:
        ETLLogger().error(f"Extraction failed: {str(e)}")
        ETLLogger().exception("Extraction error details:")
        return 1

    if debug_mode:
        df = df.head(20)

    # ==================== MANIPULATION ====================
    ETLLogger().info("")
    ETLLogger().info("=" * 80)
    ETLLogger().info("STAGE 2: MANIPULATION")
    ETLLogger().info("=" * 80)

    try:
        manipulator = DataManipulation()
        df = manipulator.manipulate(df)

        ETLLogger().info(f"Manipulation complete: {len(df)} records")
    except Exception as e:
        ETLLogger().error(f"Manipulation failed: {str(e)}")
        ETLLogger().exception("Manipulation error details:")
        return 1

    # ==================== LOAD WITH PARTITIONING ====================
    ETLLogger().info("")
    ETLLogger().info("=" * 80)
    ETLLogger().info("STAGE 3: LOAD & PARTITIONING")
    ETLLogger().info("=" * 80)

    try:
        DAL.load_data(df)

        ETLLogger().info("Load complete: data saved with medians and partitions in single operation")
    except Exception as e:
        ETLLogger().error(f"Load failed: {str(e)}")
        ETLLogger().exception("Load error details:")
        return 1

    # ==================== COMPLETION ====================
    ETLLogger().info("")
    ETLLogger().info("=" * 80)
    ETLLogger().info("✓ ETL PIPELINE COMPLETED SUCCESSFULLY")
    ETLLogger().info("=" * 80)
    ETLLogger().info(f"Log file saved to: {ETLLogger().get_log_file()}")

    ETLLogger().close()
    return 0


if __name__ == "__main__":
    exit(main())
