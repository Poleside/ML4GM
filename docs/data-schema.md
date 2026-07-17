# Annual glacier data schema

ML4GM reads a rectangular CSV table with one row per glacier and target year.
Column names may be configured, but the defaults below are recommended.

| Field | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `rgiid` | non-empty string | dimensionless identifier | Stable glacier identifier, normally linked to the documented RGI version |
| `year` | integer | calendar year | Target year represented by the row |
| `dhdt` | finite float | metres per year (`m yr⁻¹`) | Annual glacier surface-elevation change target |
| `Area` | finite float | square kilometres (`km²`) | Glacier area from the documented inventory/version |
| `Zmed` | finite float | metres above sea level (`m a.s.l.`) | Median glacier surface elevation |
| `t2m` | finite float | degrees Celsius (`°C`) or kelvin (`K`) | Annual temperature aggregate; the chosen convention must be recorded |
| `tp` | finite float | millimetres water equivalent (`mm w.e.`) or metres (`m`) | Annual total precipitation; the chosen convention must be recorded |

Static feature names are source-specific. Common examples include area,
minimum/median/maximum elevation, slope, aspect, latitude, and longitude. Every
prepared dataset must document the definition, source version, unit, spatial
aggregation, and missing-value treatment of each feature.

## Monthly sequence naming

Seasonal models use `<month>_<variable>` columns, where month is an integer from
1 through 12. For example, `1_t2m` is January near-surface temperature and
`12_tp` is December total precipitation. All 12 months required by a selected
variable must be present, in calendar order, and use one consistent unit and
aggregation convention.

## Validation rules

The package rejects missing identifiers, years, or targets; non-integer years;
duplicate glacier-year rows; non-numeric targets; non-finite numeric values; and
tables without a feature column. Duplicate observations must be resolved by an
explicit, scientifically justified aggregation step before validation.

The synthetic sample follows this shape only. Its values are generated and
cannot be interpreted as observations.
