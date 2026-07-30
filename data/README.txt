Input catalog

The pipeline reads data/Final_.csv, an ISC Bulletin export for the region
25-29 N, 54-58 E covering 1998-2023. The file name is set by RAW_FILENAME in
zm/config.py.

Columns recognised by step 0, matched case-insensitively through the alias
table in zm/catalog.py:

  DATE        origin date
  TIME        origin time
  LAT         latitude, decimal degrees
  LON         longitude, decimal degrees
  DEPTH       depth, km
  MAG         magnitude
  MAG_TYPE    magnitude type, for example mb, ML, Mw
  AUTHOR      reporting agency, optional
  EVENT_TYPE  event type, optional

DATE and TIME may also be supplied as a single combined column. If MAG_TYPE is
absent, every event is recorded as "unknown" and the magnitude-type census in
outputs/cleaning_report.json carries no information; the reported analysis
requires this column, because the mixture of mb and ML determinations is
quantified from it.

Step 0 writes the processed catalog to data/catalog_clean.csv.
