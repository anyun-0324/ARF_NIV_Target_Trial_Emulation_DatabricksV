"""Final-output validation for the CLIF ICU/IMC notebook.

8 covariate checks and 8 exposure/outcome checks return True/False.
Statistical reports are descriptive and have no automatic pass/fail cutoffs.

Before exposure truncates and renumbers vary_df, preserve its final covariates:

    vary_covariates_df = vary_df

After assembled_df is complete:

    from final_dataset_validation import validate_final_datasets
    validation = validate_final_datasets(
        baseline_df, vary_covariates_df, hospital_block_key_obj,
        outcomes_df, assembled_df,
    )
    display(validation["engineering"])
    display(validation["missingness"])
    display(validation["covariate_numeric"])
    display(validation["covariate_categories"])
    display(validation["exposure_outcome_numeric"])
    display(validation["exposure_outcome_categories"])
    display(validation["clinical_variables"])

baseline_df includes the treatment fields added by the exposure section.
vary_df still has -2, 0, ..., 22; the exported post-exposure vary_df does not.
hospital_block_key_obj is already restricted to the sampled final cohort.

Each final DataFrame is collected once. Pandas does the checks and summaries;
Spark computes timestamp expectations in the session time zone before collection.
This is for small final results, not the raw CLIF tables. Violations count rule
failures, not necessarily distinct patients. Fix engineering failures before
interpreting statistics.

Final-only checks cannot prove correct source selection, carried-value provenance,
or exact R equivalence. The current code permits some assignments before t_0;
these are reported, not redefined as invalid. Events exactly at t_0 can have no
analysis periods. Added analysis periods intentionally lack hospitalization_id.
The older QC attachment reverses ICU/stepdown and mentions obsolete 60-day
outcomes. This file follows the current code: ICU=1, stepdown=0, 28-day outcomes.
"""

import numpy as np
import pandas as pd
from pyspark.sql import functions as F


### Define validation settings
# --- Use column names and domains from the current notebook ---
KEYS = ["patient_id", "hospital_block_id"]
PERIOD_KEYS = [*KEYS, "time_block"]
SOFA_COLUMNS = ["sofa_brain", "sofa_liver", "sofa_coagulation", "sofa_kidney"]
FILL_COLUMNS = [*SOFA_COLUMNS, "code_status_category", "device_category"]
FLAG_COLUMNS = ["antihypertensive_drip", "naloxone"]
BASELINE_FLAGS = ["pulmonary_vasc", "renal_failure", "liver_disease", "metastatic_cancer"]
BASELINE_NUMERIC = ["age", "albumin_baseline", "elixhauser_index", "elixhauser_count"]
VARYING_NUMERIC = [
    "avg_temp", "avg_map", "avg_hr", "avg_sodium", "avg_potassium", "avg_wbc",
    "avg_bicarb", "avg_lactate", "avg_pco2", "avg_ph", "avg_sf", "avg_pf",
]
EVENT_COLUMNS = {
    "death_by_28d": "death_dttm_by_28d",
    "hospice_transition": "hospice_discharge_dttm_by_28d",
    "death_or_hospice_by_28d": "death_or_hospice_dttm_by_28d",
}
EVENT_TIME = "death_or_hospice_dttm_by_28d"
FOLLOWUP_TIME = "discharge_death_hospice_dttm_by_28d"
CENSOR_TIME = "discharge_censor_dttm_by_28d"
HOUR_US = 3_600_000_000
CHECK_NAMES = {
    "covariates": [
        "Patient membership", "Unique output keys", "Complete 13-period grid",
        "Required fields present", "Selected carry-forward fills",
        "Correct period timestamps", "Patient/block/encounter associations",
        "Score and flag domains",
    ],
    "exposure_outcomes": [
        "Output keys, anchors, and patient retention",
        "Treatment path and assignment consistency",
        "Fixed and varying treatment consistency", "Analysis period grid and boundaries",
        "Patient event flags, dates, and composite", "Alive-discharge censoring",
        "Follow-up stop time and truncation", "Event allocation and outcome reconciliation",
    ],
}


### Define small validation helpers
def _missing(values):
    return values.isna() | values.astype("string").str.strip().eq("").fillna(False)


def _same(left, right):
    return (left.eq(right).fillna(False) | (left.isna() & right.isna())).astype(bool)


def _pairs(frame, columns=KEYS):
    return set(frame[columns].itertuples(index=False, name=None))


def _record(results, family, number, rules, detail=""):
    failed = {name: int(count) for name, count in rules.items() if int(count)}
    results.append({
        "family": family, "check": number,
        "description": CHECK_NAMES[family][number - 1],
        "passed": not failed, "violations": sum(failed.values()),
        "failed_rules": "; ".join(f"{name}: {count}" for name, count in failed.items()),
        "detail": detail,
    })


def _collect_final(df, expected=None):
    # Epoch microseconds preserve instants and avoid pandas' year-2262 timestamp limit.
    frame = df.withColumns(expected) if expected else df
    return frame.select(*[
        F.unix_micros(F.col(f"`{name}`").cast("timestamp")).alias(name)
        if dtype.startswith("timestamp") else F.col(f"`{name}`")
        for name, dtype in frame.dtypes
    ]).toPandas()


### Check baseline and varying covariates
# --- Return exactly eight covariate integrity results ---
def _covariate_checks(baseline, varying, key):
    results = []

    # 1. Compare patient membership, not just patient counts.
    patient_sets = [set(df.loc[~_missing(df.patient_id), "patient_id"])
                    for df in (baseline, varying, key)]
    _record(results, "covariates", 1, {
        "empty_baseline": not patient_sets[0],
        "varying_patient_difference": len(patient_sets[0] ^ patient_sets[1]),
        "key_patient_difference": len(patient_sets[0] ^ patient_sets[2]),
    })

    # 2. Duplicate keys detect both exact and conflicting duplicate output rows.
    _record(results, "covariates", 2, {
        "duplicate_baseline_patient": baseline.duplicated("patient_id").sum(),
        "duplicate_baseline_block": baseline.duplicated("hospital_block_id").sum(),
        "duplicate_person_period": varying.duplicated(PERIOD_KEYS).sum(),
        "duplicate_key_mapping": key.duplicated([*KEYS, "hospitalization_id"]).sum(),
    })

    # 3. Require at most 13 blocks per patient
    groups = varying.groupby(KEYS, dropna=False)
    _record(results, "covariates", 3, {
        "wrong_period_count": groups.size().gt(13).sum(),
        "wrong_distinct_period_count": groups.time_block.nunique().gt(13).sum(),
        "unexpected_period_label": (~varying["time_block"].isin([i for i in range(0, 25, 2)])).sum(),
    })

    # 4. Required identifiers and times cannot be null, NaN, or blank.
    required = {
        "baseline": (baseline, [*KEYS, "t_0"]),
        "varying": (varying, [*PERIOD_KEYS, "hospitalization_id", "t_0", "block_start", "block_end"]),
        "key": (key, [*KEYS, "hospitalization_id"]),
    }
    _record(results, "covariates", 4, {
        f"{name}_missing_{column}": _missing(frame[column]).sum()
        for name, (frame, columns) in required.items() for column in columns
    })

    # 5. New observations can change values; renewed missing gaps violate filling.
    ordered = varying.sort_values(PERIOD_KEYS)
    rules = {}
    for column in FILL_COLUMNS:
        missing = _missing(ordered[column])
        seen = (~missing).groupby([ordered[c] for c in KEYS], dropna=False).cummax()
        rules[column] = (missing & seen).sum()
        if column in SOFA_COLUMNS:
            measured = pd.to_numeric(ordered[column], errors="coerce").notna()
            measured_any = measured.groupby([ordered[c] for c in KEYS], dropna=False).transform("any")
            rules[f"{column}_mixed_not_measured"] = (
                ordered[column].eq("not measured").fillna(False) & measured_any
            ).sum()

            
    _record(results, "covariates", 5, rules,
            "SF/PF excluded. Tests gaps, not the provenance of carried values.")

    # 6. Match the original interval expressions in the Spark session time zone.
    _record(results, "covariates", 6, {
        "incorrect_block_start": (~_same(varying.block_start, varying.__expected_start)).sum(),
        "incorrect_block_end": (~_same(varying.block_end, varying.__expected_end)).sum(),
        "nonpositive_interval": (~(varying.block_start < varying.block_end)).sum(),
    }, "Timestamp expectations retain Spark's daylight-saving behavior.")

    # 7. Preserve the baseline anchor and the hospital-block key associations.
    anchors = [*KEYS, "t_0"]
    mappings = [*KEYS, "hospitalization_id"]
    _record(results, "covariates", 7, {
        "wrong_baseline_anchor": len(_pairs(varying, anchors) - _pairs(baseline, anchors)),
        "wrong_hospitalization_mapping": len(_pairs(varying, mappings) - _pairs(key, mappings)),
        "key_maps_hospitalization_to_multiple_blocks": (
            key.groupby(["patient_id", "hospitalization_id"], dropna=False)
            .hospital_block_id.nunique().gt(1).sum()
        ),
    })

    # 8. Check output domains without revalidating global clinical thresholds.
    rules = {}
    for column in SOFA_COLUMNS:
        values = varying[column]
        valid = values.isna() | values.eq("not measured") | pd.to_numeric(values, errors="coerce").isin(range(5))
        rules[column] = (~valid.fillna(False)).sum()
    for column in FLAG_COLUMNS:
        rules[column] = (~varying[column].isin([True, False])).sum()
    for column in BASELINE_FLAGS:
        rules[column] = (~(baseline[column].isna() | baseline[column].isin([0, 1]))).sum()
    count = pd.to_numeric(baseline.elixhauser_count, errors="coerce")
    index = pd.to_numeric(baseline.elixhauser_index, errors="coerce")
    rules["elixhauser_count"] = (~(count.between(0, 29) & count.mod(1).eq(0))).sum()
    rules["elixhauser_index"] = (~(np.isfinite(index) & index.mod(1).eq(0))).sum()
    _record(results, "covariates", 8, rules,
            "SOFA=0–4/not measured/null; medication flags=boolean; negative Elixhauser scores allowed.")
    return pd.DataFrame(results)


### Check exposure and outcomes
# --- Return exactly eight exposure/outcome integrity results ---
def _exposure_outcome_checks(baseline, varying, outcomes, assembled):
    results = []

    # Avoid expanding validation joins if an output has already failed uniqueness.
    base = baseline.drop_duplicates(KEYS).copy()
    outcome_lookup = outcomes.drop_duplicates(KEYS)
    outcome_columns = [*EVENT_COLUMNS, *EVENT_COLUMNS.values(), "discharge_dttm", "discharge_category", CENSOR_TIME, FOLLOWUP_TIME]
    patient = base.drop(columns=[c for c in outcome_columns if c in base]).merge(
        outcome_lookup.drop(columns="t_0"), on=KEYS, how="left",
    )
    placed = patient.treatment.isin([0, 1])
    patient["__stop"] = patient[FOLLOWUP_TIME].where(placed, patient.time_to_assignment)
    start = varying.loc[varying.time_block.eq(-2), [*KEYS, "block_start"]].drop_duplicates(KEYS)
    patient = patient.merge(start.rename(columns={"block_start": "__pre_start"}), on=KEYS, how="left")
    # Assignment in [-2,0) removes the only retained period when that period is dropped.
    loses_seed = (patient.time_to_assignment >= patient.__pre_start) & (patient.time_to_assignment < patient.t_0)
    expected_patients = _pairs(patient.loc[(patient.__stop > patient.t_0) & ~loses_seed])

    # 1. Patient-level outputs agree; analysis retains patients with usable follow-up.
    outcome_anchors = outcomes.merge(base[[*KEYS, "t_0"]], on=KEYS, how="left", suffixes=("", "__baseline"))
    rules = {
        "duplicate_outcome_patient": outcomes.duplicated("patient_id").sum(),
        "duplicate_analysis_period": assembled.duplicated(PERIOD_KEYS).sum(),
        "outcome_patient_block_difference": len(_pairs(base) ^ _pairs(outcomes)),
        "outcome_anchor_difference": (~_same(outcome_anchors.t_0, outcome_anchors.t_0__baseline)).sum(),
        "analysis_patient_block_difference": len(expected_patients ^ _pairs(assembled)),
    }
    for name, frame, columns in [
        ("outcome", outcomes, [*KEYS, "t_0"]),
        ("analysis", assembled, [*PERIOD_KEYS, "t_0", "block_start", "block_end"]),
    ]:
        rules.update({f"{name}_missing_{c}": _missing(frame[c]).sum() for c in columns})
    _record(results, "exposure_outcomes", 1, rules,
            "Accounts for zero follow-up and removal of the original [-2,0) period.")

    # 2. ICU=1 and stepdown=0, determined by the first non-ED transition.
    expected_treatment = pd.Series(np.nan, index=base.index)
    first_location = pd.Series("", index=base.index, dtype="string")
    invalid_paths = 0
    for index, value in base.treatment_transition_path.items():
        parts = value.split(" -> ") if isinstance(value, str) and value else []
        invalid_paths += int(
            not parts or any(not p or p != p.strip().lower() for p in parts)
            or any(p in {"procedural", "radiology", "dialysis", "other"} for p in parts)
            or any(a == b for a, b in zip(parts, parts[1:]))
        )
        if parts:
            first_location.loc[index] = parts[0]
            first_non_ed = next((p for p in parts if p != "ed"), None)
            if parts[0] == "ed":
                expected_treatment.loc[index] = {"icu": 1, "stepdown": 0}.get(first_non_ed, np.nan)
    _record(results, "exposure_outcomes", 2, {
        "invalid_transition_path": invalid_paths,
        "path_treatment_mismatch": (~_same(base.treatment, expected_treatment)).sum(),
        "randomization_mismatch": (~base.randomization.eq(base.treatment.isin([0, 1]).astype(int)).fillna(False)).sum(),
        "missing_assignment": base.time_to_assignment.isna().sum(),
        "assignment_after_24h": (base.time_to_assignment > base.__expected_24h).sum(),
        "non_ed_start_not_censored_at_t0": (first_location.ne("ed") & ~_same(base.time_to_assignment, base.t_0)).sum(),
    }, "Before-t_0 assignments are reported statistically because the supplied ETL permits them.")

    # Attach expected values without replacing any actual analysis fields.
    reference_columns = ["t_0", "treatment", "treatment_transition_path", "time_to_assignment", FOLLOWUP_TIME, EVENT_TIME, "death_or_hospice_by_28d", "__stop"]
    reference = patient[[*KEYS, *reference_columns]].rename(columns={c: f"__ref_{c}" for c in reference_columns})
    analysis = assembled.merge(reference, on=KEYS, how="left")
    analysis_placed = analysis.__ref_treatment.isin([0, 1])

    # 3. Preserve fixed assignment; switch the varying ICU indicator by interval end.
    expected_varying = pd.Series(np.nan, index=analysis.index)
    expected_varying.loc[analysis.__ref_treatment.eq(0)] = 0
    icu = analysis.__ref_treatment.eq(1)
    expected_varying.loc[icu] = (analysis.loc[icu, "__ref_time_to_assignment"] <= analysis.loc[icu, "block_end"]).astype(int)
    _record(results, "exposure_outcomes", 3, {
        "baseline_anchor_changed": (~_same(analysis.t_0, analysis.__ref_t_0)).sum(),
        "fixed_treatment_changed": (~_same(analysis.treatment_indicator, analysis.__ref_treatment)).sum(),
        "transition_path_changed": (~_same(analysis.treatment_transition_path, analysis.__ref_treatment_transition_path)).sum(),
        "treatment_time_changed": (~_same(analysis.time_to_treatment, analysis.__ref_time_to_assignment.where(analysis_placed))).sum(),
        "treatment_censor_changed": (~_same(analysis.time_to_censor_from_treatment, analysis.__ref_time_to_assignment.where(~analysis_placed))).sum(),
        "varying_treatment_mismatch": (~_same(analysis.treatment_varying, expected_varying)).sum(),
    })

    # 4. Require a complete prefix of periods 2,...,24,48,...,672.
    ordered = analysis.sort_values(PERIOD_KEYS)
    grouped = ordered.groupby(KEYS, dropna=False)
    position = grouped.cumcount()
    labels = [*range(2, 25, 2), *range(48, 673, 24)]
    previous_end = grouped.block_end.shift().where(position.gt(0), ordered.t_0)
    _record(results, "exposure_outcomes", 4, {
        "incorrect_period_sequence": ordered.time_block.map({v: i for i, v in enumerate(labels)}).ne(position).sum(),
        "incorrect_period_end": (~_same(ordered.block_end, ordered.__expected_end)).sum(),
        "noncontiguous_or_wrong_first_start": (~_same(ordered.block_start, previous_end)).sum(),
        "nonpositive_interval": (~(ordered.block_start < ordered.block_end)).sum(),
    }, "The assembled analysis is not expected to retain 13 rows per patient.")

    # 5. Event dates and flags agree and form the earliest-event composite.
    rules = {}
    for indicator, timestamp in EVENT_COLUMNS.items():
        event = patient[timestamp]
        rules[f"{indicator}_flag_date_mismatch"] = (~patient[indicator].eq(event.notna().astype(int)).fillna(False)).sum()
        rules[f"{indicator}_outside_28d"] = (event.notna() & ~event.between(patient.t_0, patient.__expected_28d)).sum()
        rules[f"{indicator}_without_placement"] = (event.notna() & ~placed).sum()
        rules[f"{indicator}_after_followup"] = (event.notna() & ~(event <= patient[FOLLOWUP_TIME])).sum()
        rules[f"{indicator}_earlyorequal_trt_assignment"] = patient.loc[(patient["treatment"].isin([0,1])), "time_to_assignment"].ge(patient.loc[ (patient["treatment"].isin([0,1])), timestamp]).sum()
    earliest = patient[["death_dttm_by_28d", "hospice_discharge_dttm_by_28d"]].min(axis=1)
    rules["composite_not_earliest_event"] = (~_same(patient[EVENT_TIME], earliest)).sum()


    _record(results, "exposure_outcomes", 5, rules,
            "Events at t_0/day 28 and simultaneous death/hospice are permitted by the supplied code.")

    # 6. Censor at the qualifying terminal discharge; an unknown category is allowed.
    category = patient.discharge_category.astype("string").str.strip().str.lower()
    valid_discharge = (
        patient.discharge_dttm.between(patient.t_0, patient.__expected_28d)
        & ~category.isin(["expired", "hospice"])
        & ~(placed & (patient.discharge_dttm < patient.time_to_assignment))
    )
    _record(results, "exposure_outcomes", 6, {
        "incorrect_alive_discharge_censor": (~_same(patient[CENSOR_TIME], patient.discharge_dttm.where(valid_discharge))).sum(),
    }, "Unplaced patients may retain this intermediate censor timestamp, matching the supplied code.")

    # 7. Use the earliest outcome stop; require exactly one analysis stop mechanism.
    expected_followup = patient[[EVENT_TIME, CENSOR_TIME, "__expected_28d"]].min(axis=1).where(placed)
    final_rows = ordered.groupby(KEYS, dropna=False).tail(1)
    _record(results, "exposure_outcomes", 7, {
        "wrong_patient_followup": (~_same(patient[FOLLOWUP_TIME], expected_followup)).sum(),
        "analysis_followup_changed": (~_same(analysis[FOLLOWUP_TIME], analysis[f"__ref_{FOLLOWUP_TIME}"])).sum(),
        "both_or_neither_stop_times": (analysis[FOLLOWUP_TIME].notna() == analysis.time_to_censor_from_treatment.notna()).sum(),
        "row_starts_at_or_after_stop": (~(analysis.block_start < analysis.__ref___stop)).sum(),
        "final_interval_does_not_contain_stop": (~((final_rows.block_start < final_rows.__ref___stop) & (final_rows.__ref___stop <= final_rows.block_end))).sum(),
    })

    # 8. Allocate each event to one (start,end] period and reconcile with outcomes_df.
    expected_event = (
        analysis[f"__ref_{EVENT_TIME}"].notna()
        & (analysis[f"__ref_{EVENT_TIME}"] > analysis.block_start)
        & (analysis[f"__ref_{EVENT_TIME}"] <= analysis.block_end)
    ).astype(int)
    counts = analysis.groupby(KEYS, dropna=False).death_or_hospice_by_28d.sum(min_count=1).rename("__event_rows")
    reconciliation = patient.merge(counts, on=KEYS, how="left")
    expected_count = (
        reconciliation.death_or_hospice_by_28d.eq(1)
        & (reconciliation[EVENT_TIME] > reconciliation.t_0)
        & pd.Series([k in expected_patients for k in reconciliation[KEYS].itertuples(index=False, name=None)], index=reconciliation.index)
    ).astype(int)
    _record(results, "exposure_outcomes", 8, {
        "analysis_event_time_changed": (~_same(analysis[EVENT_TIME], analysis[f"__ref_{EVENT_TIME}"])).sum(),
        "wrong_event_interval": (~analysis.death_or_hospice_by_28d.fillna(0).eq(expected_event)).sum(),
        "event_count_mismatch": (~reconciliation.__event_rows.fillna(0).eq(expected_count)).sum(),
    }, "Zero analysis events are expected for an event exactly at t_0 with no follow-up periods.")
    return pd.DataFrame(results)


### Describe statistical distributions
# --- Separate nulls, blank strings, and explicit not-measured labels ---
def _missingness(frame, dataset, original_columns):
    rows = []
    for column in original_columns:
        values = frame[column]
        text = values.astype("string").str.strip().str.lower()
        nulls = int(values.isna().sum())
        blanks = int(text.eq("").fillna(False).sum())
        labels = int(text.eq("not measured").fillna(False).sum())
        unavailable = nulls + blanks + labels
        rows.append({
            "dataset": dataset, "variable": column, "rows": len(frame),
            "null_or_nan": nulls, "blank": blanks, "not_measured": labels,
            "unavailable": unavailable,
            "missing_pct": 100 * nulls / len(frame) if len(frame) else np.nan,
            "unavailable_pct": 100 * unavailable / len(frame) if len(frame) else np.nan,
        })
    return pd.DataFrame(rows)


def _summary_groups(frame, by_period=False):
    # Overall varying summaries weight periods; per-period summaries compare patients.
    periods = [("all", frame)]
    if by_period:
        periods.extend(frame.groupby("time_block", dropna=False, sort=True))
    for period, period_df in periods:
        yield period, "all", period_df
        for group, group_df in period_df.groupby("__treatment_group", dropna=False):
            yield period, group, group_df


def _numeric_summary(frame, columns, dataset, by_period=False):
    rows = []
    for period, group, group_df in _summary_groups(frame, by_period):
        for column in columns:
            if column not in group_df:
                continue
            raw = group_df[column]
            numeric = pd.to_numeric(raw, errors="coerce")
            finite = numeric.notna() & np.isfinite(numeric)
            values = numeric.loc[finite].astype(float)
            quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            unavailable = _missing(raw) | raw.astype("string").str.strip().str.lower().eq("not measured").fillna(False)
            rows.append({
                "dataset": dataset, "time_block": period, "treatment_group": group,
                "variable": column, "n_rows": len(group_df), "n_numeric": len(values),
                "n_unavailable": int(unavailable.sum()),
                "n_unparseable": int((~unavailable & numeric.isna()).sum()),
                "n_infinite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
                "mean": values.mean(), "sd": values.std(), "min": values.min(),
                "p01": quantiles.loc[0.01], "p25": quantiles.loc[0.25],
                "median": quantiles.loc[0.5], "p75": quantiles.loc[0.75],
                "p99": quantiles.loc[0.99], "max": values.max(),
            })
    return pd.DataFrame(rows)


def _category_summary(frame, columns, dataset, by_period=False):
    rows = []
    for period, group, group_df in _summary_groups(frame, by_period):
        for column in columns:
            if column not in group_df:
                continue
            values = group_df[column].astype("string")
            values = values.mask(values.str.strip().eq(""), "<blank>").fillna("<missing>")
            for level, count in values.value_counts(dropna=False).items():
                rows.append({
                    "dataset": dataset, "time_block": period, "treatment_group": group,
                    "variable": column, "level": level, "n": int(count),
                    "denominator": len(group_df), "percent": 100 * count / len(group_df),
                })
    return pd.DataFrame(rows)


def _clinical_variables(baseline, varying):
    # Descriptive review priorities, not a causal adjustment set or new clinical cutoffs.
    groups = [
        ("baseline", ["age"], "high", "Age and baseline vulnerability"),
        ("baseline", ["elixhauser_index", "elixhauser_count", *BASELINE_FLAGS], "high", "Comorbidity burden and major chronic organ disease"),
        ("baseline", ["albumin_baseline"], "supporting", "Baseline illness/nutritional context; nonspecific"),
        ("varying", ["avg_ph", "avg_pco2", "avg_sf", "avg_pf", "device_category"], "high", "Gas exchange, acid-base status, and respiratory support"),
        ("varying", ["avg_map", "avg_hr", "avg_lactate"], "high", "Hemodynamic and perfusion context"),
        ("varying", SOFA_COLUMNS, "high", "Separate organ-dysfunction components; examine category distributions"),
        ("varying", ["code_status_category"], "high", "Care limitations"),
        ("varying", ["avg_temp", "avg_wbc", "avg_sodium", "avg_potassium", "avg_bicarb", *FLAG_COLUMNS], "supporting", "Inflammatory, metabolic, electrolyte, and medication context"),
    ]
    return pd.DataFrame([
        {"dataset": dataset, "variable": column, "priority": priority,
         "reason": reason, "present": column in (baseline if dataset == "baseline" else varying)}
        for dataset, columns, priority, reason in groups for column in columns
    ])


def _statistical_reports(baseline, varying, outcomes, assembled):
    baseline = baseline.copy()
    baseline["__treatment_group"] = baseline.treatment.map({1: "icu", 0: "stepdown"}).fillna("unplaced")
    lookup = baseline[[*KEYS, "__treatment_group", "time_to_assignment", "__expected_24h", "__expected_28d"]].drop_duplicates(KEYS)
    varying = varying.merge(lookup[[*KEYS, "__treatment_group"]], on=KEYS, how="left")
    outcomes = outcomes.merge(lookup, on=KEYS, how="left")
    assembled = assembled.merge(lookup[[*KEYS, "__treatment_group"]], on=KEYS, how="left")

    # Patient-level exposure/outcome counts avoid weighting patients by follow-up length.
    baseline["assignment_hours_from_t0"] = (baseline.time_to_assignment - baseline.t_0) / HOUR_US
    baseline["assignment_before_t0"] = baseline.time_to_assignment < baseline.t_0
    analysis_patients = _pairs(assembled)
    baseline["has_analysis_periods"] = [k in analysis_patients for k in baseline[KEYS].itertuples(index=False, name=None)]
    baseline["assignment_status"] = np.select(
        [baseline.treatment.eq(1), baseline.treatment.eq(0),
         baseline.time_to_assignment.isna(), baseline.time_to_assignment.eq(baseline.__expected_24h)],
        ["icu", "stepdown", "missing assignment time", "unplaced at 24h"],
        default="unplaced before 24h",
    )
    for name, timestamp in [
        ("followup_hours", FOLLOWUP_TIME), ("death_hours", "death_dttm_by_28d"),
        ("hospice_hours", "hospice_discharge_dttm_by_28d"),
        ("death_or_hospice_hours", EVENT_TIME),
    ]:
        outcomes[name] = (outcomes[timestamp] - outcomes.t_0) / HOUR_US
    outcomes["followup_days"] = outcomes.followup_hours / 24
    outcomes["followup_reason"] = np.select(
        [outcomes.__treatment_group.eq("unplaced"), outcomes[FOLLOWUP_TIME].isna(),
         outcomes.death_or_hospice_by_28d.eq(1),
         outcomes[CENSOR_TIME].notna() & outcomes[CENSOR_TIME].eq(outcomes[FOLLOWUP_TIME]),
         outcomes[FOLLOWUP_TIME].eq(outcomes.__expected_28d)],
        ["unplaced", "missing followup", "death or hospice", "alive discharge", "day 28"],
        default="unexplained followup end",
    )
    placed_outcomes = outcomes.loc[outcomes.__treatment_group.isin(["icu", "stepdown"])]

    return {
        "covariate_numeric": pd.concat([
            _numeric_summary(baseline, BASELINE_NUMERIC, "baseline_df"),
            _numeric_summary(varying, VARYING_NUMERIC, "vary_df", by_period=True),
        ], ignore_index=True),
        "covariate_categories": pd.concat([
            _category_summary(baseline, [*BASELINE_FLAGS, "year"], "baseline_df"),
            _category_summary(varying, [*FILL_COLUMNS, *FLAG_COLUMNS], "vary_df", by_period=True),
        ], ignore_index=True),
        "exposure_outcome_numeric": pd.concat([
            _numeric_summary(baseline, ["assignment_hours_from_t0"], "exposure"),
            _numeric_summary(placed_outcomes, ["followup_hours", "followup_days", "death_hours", "hospice_hours", "death_or_hospice_hours"], "outcomes_placed"),
        ], ignore_index=True),
        "exposure_outcome_categories": pd.concat([
            _category_summary(baseline, ["treatment", "randomization", "treatment_transition_path", "assignment_status", "assignment_before_t0", "has_analysis_periods"], "exposure"),
            _category_summary(placed_outcomes, list(EVENT_COLUMNS), "outcomes_placed"),
            _category_summary(outcomes, ["followup_reason"], "outcomes_df"),
            _category_summary(assembled, ["treatment_varying", "death_or_hospice_by_28d"], "assembled_df", by_period=True),
        ], ignore_index=True),
        "clinical_variables": _clinical_variables(baseline, varying),
        "notes": pd.DataFrame({"note": [
            "Missingness covers every original column except unit_location. Event dates are missing when no event is recorded; unplaced patients lack treatment and outcome-follow-up times.",
            "Newly extended assembled periods intentionally lack hospitalization_id. The nonmissing-ID check applies to the original covariate scaffold.",
            "Overall varying summaries weight person-period rows; per-period summaries compare patients. Outcome percentages use placed patients and are crude recorded-event proportions, not censoring-adjusted 28-day risks.",
            "SOFA 'not measured' is separate from numeric scores. avg_ph is parsed only for summaries. SF/PF resets are excluded from fill checks.",
            "Follow-up hours/days measure elapsed time; the administrative end is 28 calendar days in the Spark session time zone.",
            "Respiratory rate is clinically relevant but absent from the supplied final columns. avg_pco2 pools arterial/venous categories and is not uniformly arterial PaCO2.",
            "Clinical priorities are a descriptive review list, not an endorsed causal adjustment set. See ERS/ATS NIV guidance and FLORALI:",
            "https://www.thoracic.org/statements/resources/cc/niv-guidelines.pdf",
            "https://doi.org/10.1056/NEJMoa1503326",
            "Final-only checks do not reconstruct raw-source selection or certify exact R/PySpark equivalence. Statistical tables require review rather than automatic pass/fail.",
        ]}),
    }


### Run final validation
# --- Collect each small final result once, then validate in pandas ---
def validate_final_datasets(baseline_df, vary_df, hospital_block_key_obj, outcomes_df, assembled_df):
    """Return the engineering and statistical report DataFrames in one dictionary.

    Data violations return False. Missing/ambiguous required columns return 16
    False rows explicitly marked not evaluated. Spark execution errors propagate.
    """
    inputs = {
        "baseline_df": baseline_df, "vary_df": vary_df,
        "hospital_block_key_obj": hospital_block_key_obj,
        "outcomes_df": outcomes_df, "assembled_df": assembled_df,
    }
    # Exclude unit location without changing the caller's DataFrames.
    inputs = {name: frame.drop("unit_location") for name, frame in inputs.items()}

    required = {
        "baseline_df": [*KEYS, "t_0", "year", *BASELINE_FLAGS, *BASELINE_NUMERIC, "treatment", "randomization", "time_to_assignment", "treatment_transition_path"],
        "vary_df": [*PERIOD_KEYS, "hospitalization_id", "t_0", "block_start", "block_end", *FILL_COLUMNS, *FLAG_COLUMNS, *VARYING_NUMERIC],
        "hospital_block_key_obj": [*KEYS, "hospitalization_id"],
        "outcomes_df": [*KEYS, "t_0", *EVENT_COLUMNS, *EVENT_COLUMNS.values(), "discharge_dttm", "discharge_category", CENSOR_TIME, FOLLOWUP_TIME],
        "assembled_df": [*PERIOD_KEYS, "hospitalization_id", "t_0", "block_start", "block_end", "treatment_indicator", "treatment_transition_path", "treatment_varying", "time_to_treatment", "time_to_censor_from_treatment", "death_or_hospice_by_28d", EVENT_TIME, FOLLOWUP_TIME],
    }
    errors = []

    for name, frame in inputs.items():
        if len(frame.columns) != len(set(frame.columns)):
            errors.append("f{name}: duplicate column names")
        if sorted(set(required[name]) - set(frame.columns)):
            errors.append(f"{name}: missing columns {absent}")
        if any(c.startswith("__expected_") for c in frame.columns):
            errors.append(f"{name}: reserved __expected_ column name")

    if errors:
        results = []
        for family in CHECK_NAMES:
            for number in range(1, 9):
                _record(results, family, number, {"input_schema_error": len(errors)},
                        "Not evaluated: " + "; ".join(errors))
        return {
            "engineering": pd.DataFrame(results),
            **{name: pd.DataFrame() for name in [
                "missingness", "covariate_numeric", "covariate_categories",
                "exposure_outcome_numeric", "exposure_outcome_categories", "clinical_variables",
            ]},
            "notes": pd.DataFrame({"note": errors}),
        }

    baseline = _collect_final(inputs["baseline_df"], {
        "__expected_24h": F.col("t_0") + F.expr("interval '24' hour"),
        "__expected_28d": F.col("t_0") + F.expr("interval '28' day"),
    })
    varying = _collect_final(inputs["vary_df"], {
        "__expected_start": F.col("t_0") + F.expr("(time_block - 2) * interval '1' hour"),
        "__expected_end": F.col("t_0") + F.expr("time_block * interval '1' hour"),
    })
    key = _collect_final(inputs["hospital_block_key_obj"].select(*required["hospital_block_key_obj"]))
    outcomes = _collect_final(inputs["outcomes_df"])
    assembled = _collect_final(inputs["assembled_df"], {
        "__expected_end": F.col("t_0") + F.expr("time_block * interval '1' hour"),
    })
    engineering = pd.concat([
        _covariate_checks(baseline, varying, key),
        _exposure_outcome_checks(baseline, varying, outcomes, assembled),
    ], ignore_index=True)
    missingness = (
        pd.concat([
            _missingness(frame, name, inputs[name].columns)
            for name, frame in [
                ("baseline_df", baseline),
                ("vary_df", varying),
                ("outcomes_df", outcomes),
                ("assembled_df", assembled),
            ]
        ], ignore_index=True)
        .loc[:, [
            "dataset", "variable", "rows",
            "null_or_nan", "blank", "unavailable", "unavailable_pct",
        ]]
        .sort_values(
            ["dataset", "unavailable_pct"],
            ascending=[True, False],
            ignore_index=True,
        )
    )
    missingness = missingness[~missingness.variable.str.startswith("first_")]
    
    return {
        "engineering": engineering, "missingness": missingness,
        **_statistical_reports(baseline, varying, outcomes, assembled),
    }
