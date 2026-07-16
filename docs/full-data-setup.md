# Full scientific data setup

ML4GM does not automate acceptance of provider terms and does not embed
undocumented download endpoints. Start with the official landing pages and
citations in [DATA_SOURCES.md](../DATA_SOURCES.md).

## 1. Acquire sources

Download each required source directly from its provider. Record the version,
access date, accepted terms, original filename, and SHA-256 checksum. Keep raw
files outside Git, for example at paths supplied at runtime:

```text
<PATH_TO_RGI_V6>
<PATH_TO_ERA5_LAND>
<PATH_TO_HUGONNET_DATA>
```

Do not proceed with a source if its licence, citation, or intended use is
unclear. In particular, verify the official repository's current redistribution
terms for the Hugonnet dataset before sharing a subset.

## 2. Build the annual table

Join glacier geometry, annual target, and climate features using documented
spatial and temporal rules. Export a CSV conforming to
[the annual schema](data-schema.md). Preserve the processing code and never
overwrite the raw downloads.

## 3. Prepare and manifest the table

Create a configuration whose `data.input` points to
`<PATH_TO_ANNUAL_GLACIER_TABLE>`, then run:

```bash
ml4gm data prepare \
  --config <PATH_TO_CONFIG> \
  --source "RGI Consortium (2017), RGI v6, doi:10.7265/4m1f-gd79" \
  --source "ERA5-Land, doi:10.24381/cds.e2161bac" \
  --source "Hugonnet et al. (2021), doi:10.1038/s41586-021-03436-z" \
  --output <PATH_TO_PREPARED_CSV> \
  --manifest <PATH_TO_MANIFEST_JSON>
```

Inspect the schema, row and glacier counts, year range, feature list,
missing-value summary, source checksums, and output checksum. A successful
manifest validates traceability; it does not grant redistribution permission.

## 4. Evaluate

Point a benchmark configuration at the prepared CSV and use a new output
directory:

```bash
ml4gm evaluate --config <PATH_TO_BENCHMARK_CONFIG> --output-dir <PATH_TO_RESULTS>
```

Archive the resolved configuration, manifest, software version, environment,
and fold-level JSON together. Follow the
[benchmark protocol](benchmark-protocol.md) and report the
[scientific limitations](scientific-limitations.md).
