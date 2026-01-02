#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple runner script to execute the ETL pipeline main function.
"""

import sys
from ETL.etl_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
