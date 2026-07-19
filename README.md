## Data Schema

This project uses a **unified schema**: every row in `ethiopia_fi_unified_data.csv`
shares the same 34 columns, and the `record_type` field determines how to
interpret the row.

| record_type | Count | Meaning | Key fields used |
|---|---|---|---|
| `observation` | 30 | A measured value from a survey, operator report, or infrastructure source | `indicator_code`, `value_numeric`, `observation_date`, `gender`, `pillar` |
| `event` | 10 | A policy, product launch, market entry, or milestone | `category`, `observation_date`, `source_name` &mdash; `pillar` is intentionally left blank |
| `target` | 3 | An official policy goal (e.g. NFIS-II) | `indicator_code`, `value_numeric`, `period_end` |

Impact relationships between events and indicators live in a **separate file**,
`Impact_sheet.csv` (14 `impact_link` records), joined back to events via
`parent_id -> record_id`. This separation is deliberate: events are not
pre-assigned to a pillar, so their effects stay evidence-based rather than
baked into the raw catalog.

`data/raw/reference_codes.csv` lists the valid values for every categorical
field (`record_type`, `category`, `pillar`, `confidence`, `source_type`, etc.)
and is used at load time to flag any value not in the approved list (see
`src/data_loader.py::validate_against_reference`).

### Why observations, events, and impact_links are split this way

- **Observations** answer "what happened to an indicator."
- **Events** answer "what happened in the market/policy environment," independent of any claimed effect.
- **Impact_links** answer "how strongly do we believe a given event moved a given indicator, and after how long" &mdash; this is where `impact_direction`, `impact_magnitude`, and `lag_months` live.

Keeping these three concerns in separate record types (rather than one flat
"event with effects" table) is what lets Task 3's event-impact modeling stay
falsifiable: an analyst can dispute or revise an impact_link's magnitude
without having to touch the underlying observation or event record.

## Data Pipeline: Raw &rarr; Processed

```
data/raw/ethiopia_fi_unified_data.csv   \
data/raw/Impact_sheet.csv                >--  src/data_loader.py  -->  validated, typed DataFrames
data/raw/reference_codes.csv            /            |
                                                       v
                                          src/analysis.py (indicator series,
                                          event-impact merges, growth rates)
                                                       |
                                                       v
                                    data/processed/ethiopia_fi_enriched.csv
                                    data/processed/impact_sheet_enriched.csv
```

1. **Load** &mdash; `src/data_loader.py` reads the three raw CSVs, parses dates,
   validates required columns are present, and checks categorical fields
   against `reference_codes.csv`. Missing files or malformed input raise
   explicit errors rather than failing silently.
2. **Enrich** &mdash; new observations, targets, and impact_links identified
   during Task 1 (documented in `data_enrichment_log.md`) are merged into
   the same in-memory DataFrames.
3. **Analyze** &mdash; `src/analysis.py` provides the reusable functions used
   throughout `notebooks/02_eda.ipynb`: national (non-disaggregated)
   indicator series extraction, growth-rate computation, and event/impact
   merging.
4. **Persist** &mdash; the final enriched DataFrames are written to
   `data/processed/` at the end of the EDA notebook, so downstream tasks
   (impact modeling, forecasting, dashboard) read from `data/processed/`
   rather than re-deriving enrichment logic from `data/raw/`.

## Running the Project

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # confirm src/ modules pass (13 tests)
jupyter notebook notebooks/02_eda.ipynb   # Kernel -> Restart & Run All, then save
```
