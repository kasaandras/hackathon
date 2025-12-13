#!/usr/bin/env python3
"""
Utility script to apply one-hot encoding to selected columns.

Example:
    python 03_encoding.py \
        --input preprocessing_output/disease_free_cleaned_data.csv \
        --columns figo_stage_2023 preoperative_staging preoperative_risk_group
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Columns that behave like categorical factors in the cleaned disease_free dataset.
DEFAULT_CATEGORICAL_COLUMNS = [
    "estadificacion_",
    "tumor_grade_preop",
    "figo_stage_2023",
    "preoperative_staging",
    "final_grade",
    "preoperative_risk_group",
    "est_pcte",
    "tec_linf_para",
    "myometrial_invasion",
    "inten_tto",
    "recidiva_exitus",
    "final_histology",
    "primary_surgery_type",
    "histological_subtype_preop",
    "sentinel_node_pathology"
]


def parse_column_args(raw_columns: Iterable[str] | None) -> List[str]:
    """Normalize column arguments to a clean list."""
    if not raw_columns:
        return []

    columns: List[str] = []
    for token in raw_columns:
        columns.extend([col.strip() for col in token.split(",") if col.strip()])
    return columns


def get_columns_to_encode(df: pd.DataFrame, requested_columns: List[str]) -> List[str]:
    """
    Filter the requested column list by the dataframe columns and warn on missing ones.
    """
    if not requested_columns:
        requested_columns = DEFAULT_CATEGORICAL_COLUMNS

    missing = [col for col in requested_columns if col not in df.columns]
    if missing:
        print(
            "⚠️  Warning: The following columns were not found and will be skipped:",
            ", ".join(missing),
        )

    valid = [col for col in requested_columns if col in df.columns]
    if not valid:
        raise ValueError("No valid columns left to encode. Check the column names.")
    return valid


def encode_dataset(
    input_path: Path,
    output_path: Path,
    columns_to_encode: List[str],
    drop_first: bool = False,
) -> None:
    """
    Load a CSV, encode the specified columns, and write the transformed data.
    """
    print(f"Loading data from {input_path} ...")
    df = pd.read_csv(input_path)

    columns = get_columns_to_encode(df, columns_to_encode)
    print(f"Encoding {len(columns)} column(s): {', '.join(columns)}")

    encoded_df = pd.get_dummies(
        df,
        columns=columns,
        drop_first=drop_first,
        prefix_sep="__",
        dtype="int64",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(output_path, index=False)

    added_cols = encoded_df.shape[1] - df.shape[1]
    print(
        f"Done. Encoded dataset saved to {output_path} "
        f"({encoded_df.shape[0]} rows, {encoded_df.shape[1]} columns | "
        f"+{added_cols} new columns)."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply one-hot encoding to selected categorical columns of a CSV file. "
            "Columns can be provided as a whitespace separated list or as a "
            "comma-separated string."
        )
    )
    parser.add_argument(
        "--input",
        default="preprocessing_output/disease_free_cleaned_data.csv",
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the encoded CSV. Defaults to '<input>_encoded.csv'.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Columns to one-hot encode. "
            "If omitted, a default list tailored to the disease_free dataset is used."
        ),
    )
    parser.add_argument(
        "--drop-first",
        action="store_true",
        help="Drop the first level for each encoded column to avoid collinearity.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = args.output
    if not output_path:
        output_path = input_path.with_name(f"{input_path.stem}_encoded{input_path.suffix}")

    columns_to_encode = parse_column_args(args.columns)

    try:
        encode_dataset(input_path, output_path, columns_to_encode, drop_first=args.drop_first)
    except Exception as exc:
        raise SystemExit(f"Encoding failed: {exc}") from exc


if __name__ == "__main__":
    main()
