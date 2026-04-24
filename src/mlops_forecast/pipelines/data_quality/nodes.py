"""Great Expectations v1 quality checks on the raw OPSD CSV.

We run a small expectation suite against the raw frame before any cleaning,
fail the pipeline loudly if any of the structural ones break, and persist
the full result to `data/08_reporting/data_quality/` so the dashboard can
display it. The "structural" expectations are schema and uniqueness; the
range expectations are softer (mostly=0.99) because the OPSD load series
has a handful of legitimate outliers we don't want to halt on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)


def run_data_quality_checks(
    df: pd.DataFrame,
    params: dict,
) -> str:
    """Run the GE v1 expectation suite and return its JSON result.

    Raises ValueError if any critical (structural) expectation fails,
    which halts the Kedro run before bad data reaches downstream nodes.
    """
    # An ephemeral context throws away its registry on exit; we don't need
    # GE's full project layout for a single batch validation.
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="raw_electricity")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")

    # The suite has to be registered with the context before we can wire it
    # into a ValidationDefinition; this caught us once with a confusing
    # ResourceFreshnessAggregateError when we didn't.
    suite_name = params["expectation_suite_name"]
    suite = gx.ExpectationSuite(name=suite_name)

    # Schema
    for col in [
        "utc_timestamp",
        "DE_load_actual_entsoe_transparency",
        "DE_wind_generation_actual",
        "DE_solar_generation_actual",
    ]:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # Null rates
    for col in ["DE_load_actual_entsoe_transparency", "DE_wind_generation_actual"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col,
                mostly=1.0 - params["max_null_fraction"],
            )
        )

    # Value ranges
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DE_load_actual_entsoe_transparency",
            min_value=params["min_load_mw"],
            max_value=params["max_load_mw"],
            mostly=0.99,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DE_wind_generation_actual",
            min_value=params["min_wind_mw"],
            max_value=params["max_wind_mw"],
            mostly=0.99,
        )
    )

    # No duplicate timestamps
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="utc_timestamp"))

    # Row count
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=8760, max_value=None)
    )

    suite = context.suites.add(suite)
    vd = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="raw_electricity_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    result = vd.run(batch_parameters={"dataframe": df})

    n_total = result.statistics["evaluated_expectations"]
    n_passed = result.statistics["successful_expectations"]
    n_failed = n_total - n_passed
    logger.info("Data quality: %d/%d expectations passed, %d failed", n_passed, n_total, n_failed)

    out_dir = Path("data/08_reporting/data_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dict = result.to_json_dict()
    report_path = out_dir / "ge_validation_result.json"
    report_path.write_text(json.dumps(result_dict, indent=2))
    logger.info("GE report saved to %s", report_path)

    # These are the structural checks. If any of them fails the dataset is
    # unusable for the rest of the pipeline, so we abort rather than try to
    # paper over it downstream.
    critical_types = {
        "expect_column_to_exist",
        "expect_column_values_to_be_unique",
        "expect_table_row_count_to_be_between",
    }
    # `expectation_config` is typed as Optional in GE; in practice it is set
    # for every result we evaluate, so the `is None` branch never fires, but
    # we narrow the type for the type checker.
    failed_critical = [
        r
        for r in result.results
        if not r.success
        and r.expectation_config is not None
        and r.expectation_config.type in critical_types
    ]
    if failed_critical:
        msgs = [
            r.expectation_config.type for r in failed_critical if r.expectation_config is not None
        ]
        raise ValueError(
            f"CRITICAL data quality expectations failed: {msgs}. "
            "Pipeline halted to prevent corrupted data reaching models."
        )

    return json.dumps(result_dict)
