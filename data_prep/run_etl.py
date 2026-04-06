"""Wrapper to run etl_simple_femr with increased CSV field size limit."""
import csv
import sys

# Increase CSV field size limit to handle large Athena fields
csv.field_size_limit(sys.maxsize)

# Now import and run the ETL
from femr.etl_pipelines.simple import etl_simple_femr_program
etl_simple_femr_program()
