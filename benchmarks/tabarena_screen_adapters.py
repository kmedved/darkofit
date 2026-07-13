"""Benchmark-only TabArena adapters for isolated representation screens.

The production adapter intentionally preserves pandas categoricals for
DarkoFit's native ordered target statistics.  The two adapters below provide
training-fold-only, target-free alternatives for the follow-on screen.  They
live under ``benchmarks`` so an exploratory representation cannot silently
become a package default.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from benchmarks.tabarena_adapter import DarkoFitModel
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_adapter import DarkoFitModel

if TYPE_CHECKING:
    import pandas as pd


REPRESENTATION_METADATA_KEY = "benchmark_representation"
ONE_HOT_MAX_CATEGORIES_PER_FEATURE = 8
ONE_HOT_MAX_OUTPUT_FEATURES = 256

# AutoGluon's compact category code -> physical attack angle. The source labels
# are strings, so its deterministic category order is lexical rather than
# numeric (for example child code 13 represents 3.0 degrees).
AIRFOIL_CHILD_CODE_VALUES = (
    0.0,
    1.5,
    11.2,
    12.3,
    12.6,
    12.7,
    15.4,
    15.6,
    17.4,
    19.7,
    2.0,
    2.7,
    22.2,
    3.0,
    3.3,
    4.0,
    4.2,
    4.8,
    5.3,
    5.4,
    6.7,
    7.2,
    7.3,
    8.4,
    8.9,
    9.5,
    9.9,
)

AIRFOIL_COLUMNS = (
    "frequency",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
    "attack-angle",
)
DIAMONDS_COLUMNS = (
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "cut",
    "color",
    "clarity",
)
MIAMI_COLUMNS = (
    "LATITUDE",
    "LONGITUDE",
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "age",
    "avno60plus",
    "month_sold",
    "structure_quality",
)
FIAT_COLUMNS = (
    "engine_power",
    "age_in_days",
    "km",
    "previous_owners",
    "lat",
    "lon",
    "model",
)
FOOD_COLUMNS = (
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Delivery_person_ID",
    "Type_of_order",
    "Type_of_vehicle",
)
HEALTHCARE_COLUMNS = ("age", "sex", "bmi", "children", "smoker", "region")
WINE_COLUMNS = (
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_color",
)
ONE_HOT_SCHEMAS = {
    FIAT_COLUMNS: ("fiat", [6]),
    DIAMONDS_COLUMNS: ("diamonds", [6, 7, 8]),
    FOOD_COLUMNS: ("food_delivery", [6, 7, 8]),
    HEALTHCARE_COLUMNS: ("healthcare", [1, 4, 5]),
    MIAMI_COLUMNS: ("miami", [12]),
    WINE_COLUMNS: ("wine", [11]),
}
DIAMONDS_ORDERS = {
    "cut": ("Fair", "Good", "Very Good", "Premium", "Ideal"),
    "color": ("J", "I", "H", "G", "F", "E", "D"),
    "clarity": ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"),
}
# AutoGluon's source-frozen ``CategoryMemoryMinimizeFeatureGenerator`` replaces
# the labels above with alphabetical compact codes before the child adapter.
# These tuples map child code -> the declared semantic rank in DIAMONDS_ORDERS;
# no rank is inferred from the compact code itself.
DIAMONDS_CHILD_CODE_RANKS = {
    "cut": (0, 1, 4, 3, 2),
    "color": (6, 5, 4, 3, 2, 1, 0),
    "clarity": (0, 7, 2, 1, 4, 3, 6, 5),
}
ORDINAL_COMPACT_DOMAINS = {
    "airfoil_attack_angle_numeric": {"attack-angle": tuple(range(27))},
    "diamonds_declared_orders": {
        "cut": tuple(range(5)),
        "color": tuple(range(7)),
        "clarity": tuple(range(8)),
    },
    "miami_avno60plus_binary": {},
}


def _category_schema_digest(columns: list[dict[str, Any]]) -> str:
    """Hash category identities without persisting raw dataset values."""
    payload = json.dumps(
        columns,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _category_identity(value: Any) -> dict[str, str]:
    value_type = type(value)
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "repr": repr(value),
    }


class ScreenNativeDarkoFitModel(DarkoFitModel):
    """Native representation with an explicit screen audit record."""

    screen_representation = "native"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        categorical_columns = [
            str(column)
            for column in X.select_dtypes(include="category").columns
        ]
        input_feature_count = int(X.shape[1])
        super()._fit(X, y, **kwargs)
        self._fit_metadata[REPRESENTATION_METADATA_KEY] = {
            "kind": self.screen_representation,
            "fit_scope": "darkofit_child_training_fold",
            "target_used_by_representation": bool(categorical_columns),
            "input_feature_count": input_feature_count,
            "output_feature_count": int(self.model.n_features_in_),
            "categorical_input_columns": categorical_columns,
        }


class _SafeCategoricalRepresentationModel(DarkoFitModel):
    """Base class for child-training-only, target-free encodings."""

    screen_representation = "abstract"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self._screen_representation_fit_active = True
        self._screen_representation_fit_calls = 0
        self._screen_representation_eval_transform_calls = 0
        self._screen_representation_eval_unknown_counts: list[int] = []
        try:
            super()._fit(X, y, **kwargs)
        finally:
            self._screen_representation_fit_active = False
        if self._screen_representation_fit_calls != 1:
            raise RuntimeError(
                "safe benchmark representation must be fitted exactly once"
            )
        metadata = self._representation_metadata()
        metadata.update(
            {
                "fit_scope": "child_training_rows_only",
                "target_used_by_representation": False,
                "fit_calls": self._screen_representation_fit_calls,
                "eval_transform_calls_during_fit": (
                    self._screen_representation_eval_transform_calls
                ),
                "eval_unknown_counts": list(
                    self._screen_representation_eval_unknown_counts
                ),
            }
        )
        self._fit_metadata[REPRESENTATION_METADATA_KEY] = metadata

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        if is_train:
            if self._screen_representation_fit_calls:
                raise RuntimeError(
                    "safe benchmark representation cannot be refitted"
                )
            self._fit_representation(X)
            self._screen_representation_fit_calls += 1
        elif not self._screen_representation_fit_calls:
            raise RuntimeError(
                "safe benchmark representation used before child training fit"
            )
        transformed, unknown_count = self._transform_representation(X)
        if not is_train and self._screen_representation_fit_active:
            self._screen_representation_eval_transform_calls += 1
            self._screen_representation_eval_unknown_counts.append(
                int(unknown_count)
            )
        return transformed

    def _fit_representation(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def _transform_representation(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        raise NotImplementedError

    def _representation_metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class SafeOrdinalDarkoFitModel(_SafeCategoricalRepresentationModel):
    """Apply only domain-predeclared ordinal/numeric interpretations.

    No order is learned from row appearance or the target.  The exact input
    schema selects one source-frozen rule set; any other schema or category
    fails closed.
    """

    screen_representation = "safe_ordinal"

    def _fit_representation(self, X: pd.DataFrame) -> None:
        import pandas as pd

        self._representation_input_columns = list(X.columns)
        columns = tuple(str(column) for column in X.columns)
        categorical = [
            str(column)
            for column in X.columns
            if isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        if columns == AIRFOIL_COLUMNS and categorical == ["attack-angle"]:
            self._representation_domain = "airfoil_attack_angle_numeric"
            self._representation_rules = {
                "attack-angle": {
                    str(code): value
                    for code, value in enumerate(AIRFOIL_CHILD_CODE_VALUES)
                }
            }
        elif columns == DIAMONDS_COLUMNS and categorical == list(DIAMONDS_ORDERS):
            self._representation_domain = "diamonds_declared_orders"
            self._representation_rules = {
                column: {
                    str(code): rank
                    for code, rank in enumerate(DIAMONDS_CHILD_CODE_RANKS[column])
                }
                for column in DIAMONDS_ORDERS
            }
        elif columns == MIAMI_COLUMNS and categorical == []:
            self._representation_domain = "miami_avno60plus_binary"
            self._representation_rules = {
                "avno60plus": {"0": 0, "0.0": 0, "1": 1, "1.0": 1}
            }
        else:
            raise RuntimeError(
                "safe ordinal screen input does not match a predeclared schema: "
                f"columns={columns!r}, categoricals={categorical!r}"
            )
        self._representation_compact_domains = ORDINAL_COMPACT_DOMAINS[
            self._representation_domain
        ]
        self._validate_compact_category_domains(X)
        self._representation_categorical_positions = [
            self._representation_input_columns.index(column)
            for column in self._representation_rules
        ]
        self._representation_observed_counts = []
        for column in self._representation_rules:
            values = X[column].astype(object)
            self._representation_observed_counts.append(int(values.nunique(dropna=True)))
        frozen_schema = {
            "domain": self._representation_domain,
            "columns": columns,
            "rules": {
                column: [[key, value] for key, value in rule.items()]
                for column, rule in self._representation_rules.items()
            },
            "compact_category_domains": {
                column: list(domain)
                for column, domain in self._representation_compact_domains.items()
            },
        }
        self._representation_schema_sha256 = hashlib.sha256(
            json.dumps(
                frozen_schema,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._representation_output_columns = list(X.columns)

    def _validate_compact_category_domains(self, X: pd.DataFrame) -> None:
        import pandas as pd

        for column, expected in self._representation_compact_domains.items():
            if not isinstance(X[column].dtype, pd.CategoricalDtype):
                raise RuntimeError(
                    f"safe ordinal {column!r} lost its categorical dtype"
                )
            observed = tuple(X[column].cat.categories.tolist())
            if observed != expected:
                raise RuntimeError(
                    f"safe ordinal {column!r} compact category domain changed: "
                    f"observed={observed!r}, expected={expected!r}"
                )

    def _transform_representation(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        if list(X.columns) != self._representation_input_columns:
            raise RuntimeError("safe ordinal input schema changed after fitting")
        self._validate_compact_category_domains(X)
        transformed = X.copy()
        for column, rule in self._representation_rules.items():
            values = X[column].astype(object)
            missing = values.isna().to_numpy(dtype=bool, copy=False)
            encoded = np.full(len(values), np.nan, dtype=np.float64)
            for index, value in enumerate(values):
                if missing[index]:
                    continue
                key = str(value)
                if key not in rule:
                    raise RuntimeError(
                        f"safe ordinal {column!r} contains undeclared value {key!r}"
                    )
                encoded[index] = rule[key]
            transformed[column] = encoded
        self._categorical_indices = []
        return transformed, 0

    def _representation_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.screen_representation,
            "domain": self._representation_domain,
            "mapping_source": "source_frozen_before_campaign",
            "input_feature_count": len(self._representation_input_columns),
            "output_feature_count": len(self._representation_output_columns),
            "categorical_input_positions": list(
                self._representation_categorical_positions
            ),
            "observed_training_category_counts": list(
                self._representation_observed_counts
            ),
            "compact_category_domains": {
                column: list(domain)
                for column, domain in self._representation_compact_domains.items()
            },
            "category_schema_sha256": self._representation_schema_sha256,
            "missing_policy": "numeric_nan",
            "unknown_policy": "fail_closed",
            "remaining_native_target_stat_positions": [],
        }


class SafeOneHotDarkoFitModel(_SafeCategoricalRepresentationModel):
    """Dense one-hot encoding learned only from child training rows.

    The screen is deliberately limited to declared low-cardinality datasets.
    Hard caps fail closed if a schema change would make dense one-hot unsafe.
    Missing values get a dedicated indicator; unseen categories are all-zero.
    """

    screen_representation = "safe_one_hot"

    def _fit_representation(self, X: pd.DataFrame) -> None:
        import pandas as pd

        self._representation_input_columns = list(X.columns)
        observed_categorical_positions = [
            index
            for index, column in enumerate(X.columns)
            if isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        schema = ONE_HOT_SCHEMAS.get(
            tuple(str(column) for column in self._representation_input_columns)
        )
        if schema is None or not set(observed_categorical_positions).issubset(
            schema[1]
        ):
            raise RuntimeError(
                "safe one-hot screen input does not match a predeclared schema"
            )
        self._representation_domain = schema[0]
        self._representation_categorical_positions = list(schema[1])
        categorical_set = set(self._representation_categorical_positions)
        self._representation_numeric_positions = [
            index for index in range(X.shape[1]) if index not in categorical_set
        ]
        self._representation_categories = {}
        self._representation_native_positions = []
        schema = []
        for position in self._representation_categorical_positions:
            values = X.iloc[:, position].astype(object)
            categories = pd.Index(pd.unique(values[values.notna()]))
            if len(categories) == 0:
                raise RuntimeError(
                    "safe one-hot screen found an empty categorical feature"
                )
            if len(categories) > ONE_HOT_MAX_CATEGORIES_PER_FEATURE:
                if not isinstance(X.iloc[:, position].dtype, pd.CategoricalDtype):
                    raise RuntimeError(
                        "safe one-hot cannot leave a high-cardinality numeric "
                        "feature on the native target-stat path"
                    )
                self._representation_native_positions.append(position)
                schema.append(
                    {
                        "position": position,
                        "mode": "native_target_statistics",
                        "categories": [
                            _category_identity(value)
                            for value in categories.tolist()
                        ],
                    }
                )
                continue
            self._representation_categories[position] = categories
            schema.append(
                {
                    "position": position,
                    "mode": "target_free_one_hot",
                    "categories": [
                        _category_identity(value) for value in categories.tolist()
                    ],
                }
            )

        if not self._representation_categories:
            raise RuntimeError("safe one-hot screen found no <=8-cardinality feature")
        output_feature_count = (
            len(self._representation_numeric_positions)
            + len(self._representation_native_positions)
            + sum(
            len(categories) + 1
            for categories in self._representation_categories.values()
        )
        )
        if output_feature_count > ONE_HOT_MAX_OUTPUT_FEATURES:
            raise RuntimeError(
                "safe one-hot output cap exceeded: "
                f"{output_feature_count} > {ONE_HOT_MAX_OUTPUT_FEATURES}"
            )
        self._representation_schema_sha256 = _category_schema_digest(schema)
        used_names = {str(column) for column in X.columns}
        output_names = []
        self._representation_output_plan = []
        for position, original_column in enumerate(self._representation_input_columns):
            if position not in categorical_set or position in self._representation_native_positions:
                output_names.append(str(original_column))
                self._representation_output_plan.append(("passthrough", position, None))
                continue
            categories = self._representation_categories[position]
            for code in range(len(categories)):
                candidate = f"__darkofit_screen_ohe_{position}_{code}"
                while candidate in used_names:
                    candidate = "_" + candidate
                used_names.add(candidate)
                output_names.append(candidate)
                self._representation_output_plan.append(("category", position, code))
            candidate = f"__darkofit_screen_ohe_{position}_missing"
            while candidate in used_names:
                candidate = "_" + candidate
            used_names.add(candidate)
            output_names.append(candidate)
            self._representation_output_plan.append(("missing", position, None))
        if len(output_names) != output_feature_count:
            raise RuntimeError("safe one-hot output schema construction failed")
        self._representation_output_columns = output_names

    def _transform_representation(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        import pandas as pd

        if list(X.columns) != self._representation_input_columns:
            raise RuntimeError("safe one-hot input schema changed after fitting")
        series = []
        unknown_total = 0
        cached_codes = {}
        for kind, position, code in self._representation_output_plan:
            if kind == "passthrough":
                series.append(X.iloc[:, position].copy())
                continue
            if position not in cached_codes:
                values = X.iloc[:, position].astype(object)
                missing = values.isna().to_numpy(dtype=bool, copy=False)
                codes = self._representation_categories[position].get_indexer(values)
                unknown = (codes < 0) & ~missing
                unknown_total += int(np.count_nonzero(unknown))
                cached_codes[position] = (codes, missing)
            codes, missing = cached_codes[position]
            if kind == "category":
                series.append(pd.Series((codes == code).astype(np.uint8), index=X.index))
            else:
                series.append(pd.Series(missing.astype(np.uint8), index=X.index))
        transformed = pd.concat(series, axis=1)
        transformed.columns = self._representation_output_columns
        if transformed.shape[1] != len(self._representation_output_columns):
            raise RuntimeError("safe one-hot transform produced the wrong width")
        native_input_set = set(self._representation_native_positions)
        self._categorical_indices = [
            output_position
            for output_position, (kind, input_position, _) in enumerate(
                self._representation_output_plan
            )
            if kind == "passthrough" and input_position in native_input_set
        ]
        return transformed, unknown_total

    def _representation_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.screen_representation,
            "domain": self._representation_domain,
            "input_feature_count": len(self._representation_input_columns),
            "output_feature_count": len(self._representation_output_columns),
            "categorical_input_positions": list(
                self._representation_categorical_positions
            ),
            "target_free_one_hot_input_positions": sorted(
                self._representation_categories
            ),
            "target_free_one_hot_category_counts": [
                len(self._representation_categories[position])
                for position in sorted(self._representation_categories)
            ],
            "remaining_native_target_stat_input_positions": list(
                self._representation_native_positions
            ),
            "remaining_native_target_stat_output_positions": list(
                self._categorical_indices
            ),
            "remaining_native_target_stats_use_target": bool(
                self._representation_native_positions
            ),
            "category_schema_sha256": self._representation_schema_sha256,
            "unknown_policy": "all_zero",
            "missing_indicator_per_categorical": True,
            "max_categories_per_feature": ONE_HOT_MAX_CATEGORIES_PER_FEATURE,
            "max_output_features": ONE_HOT_MAX_OUTPUT_FEATURES,
        }


__all__ = [
    "AIRFOIL_CHILD_CODE_VALUES",
    "DIAMONDS_CHILD_CODE_RANKS",
    "DIAMONDS_ORDERS",
    "ONE_HOT_MAX_CATEGORIES_PER_FEATURE",
    "ONE_HOT_MAX_OUTPUT_FEATURES",
    "ORDINAL_COMPACT_DOMAINS",
    "REPRESENTATION_METADATA_KEY",
    "SafeOneHotDarkoFitModel",
    "SafeOrdinalDarkoFitModel",
    "ScreenNativeDarkoFitModel",
]
