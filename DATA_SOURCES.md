# Data sources and rights

ML4GM's Apache-2.0 license covers project-authored code only. Every researcher
is responsible for complying with the current terms shown by each data provider.
The table records authoritative landing pages verified on 2026-07-17; provider
terms can change, so check them again when acquiring or redistributing data.

| Source | Version | Purpose | Official URL/DOI | Access | License/terms | Redistribution | Required citation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Randolph Glacier Inventory (RGI) | Version 6, NSIDC-0770 | Glacier identifiers, outlines, and static geometry | [NSIDC landing page](https://nsidc.org/data/nsidc-0770/versions/6), [doi:10.7265/4m1f-gd79](https://doi.org/10.7265/4m1f-gd79) | Obtain from NSIDC and record access date | NSIDC's current data-use and citation conditions apply; the landing page requires citation | Do not commit upstream archives or derived geometry until the applicable NSIDC/GLIMS redistribution terms have been reviewed for the proposed artifact | RGI Consortium (2017), *Randolph Glacier Inventory — A Dataset of Global Glacier Outlines*, Version 6, NSIDC-0770, doi:10.7265/4m1f-gd79; describe the subset and access date |
| ERA5-Land hourly data | Continuously updated dataset, first published 2019 | Monthly or annual near-surface climate predictors | [CDS landing page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land), [doi:10.24381/cds.e2161bac](https://doi.org/10.24381/cds.e2161bac) | CDS account and acceptance of the displayed licence may be required | The CDS dataset page displays a **CC-BY** licence; use the current licence text and attribution instructions shown at access time | Redistribution is governed by the accepted CDS CC-BY licence and attribution requirements; preserve provenance and do not imply endorsement | Copernicus Climate Change Service, *ERA5-Land hourly data from 1950 to present*, doi:10.24381/cds.e2161bac, with the CDS-requested attribution and access date |
| Hugonnet et al. glacier elevation and mass change | 2000–2019 products associated with the 2021 article | `dhdt` targets and validation products | [Nature article](https://doi.org/10.1038/s41586-021-03436-z), [official data record doi:10.6096/13](https://doi.org/10.6096/13) | Follow the article's Data availability link to the official repository | The article states that data are publicly available, but that statement alone is not a redistribution licence | **Before any real subset is committed, verify the exact dataset licence and redistribution terms displayed by the official data repository for doi:10.6096/13.** Until that review is recorded, keep these files outside Git | Hugonnet, R. et al. (2021), “Accelerated global glacier mass loss in the early twenty-first century,” *Nature* 592, 726–731, doi:10.1038/s41586-021-03436-z; also cite the data record doi:10.6096/13 |

## Repository data policy

The repository includes only a generated synthetic table under `data/sample/`.
It does not contain RGI geometry, ERA5-Land values, or Hugonnet observations.
Raw downloads, processed scientific tables, trained weights, and run outputs
must remain ignored unless maintainers document source, version, checksums,
required citation, licence, and redistribution permission in a reviewed change.

Prepared datasets should include a manifest produced by ML4GM. A manifest is
provenance metadata, not proof that redistribution is authorized.
