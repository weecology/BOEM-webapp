from pathlib import Path
from typing import Optional

import pandas as pd


ANNOTATION_COLUMNS = [
    "image_id",
    "original_label",
    "new_label",
    "set",
    "timestamp",
    "user",
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected annotation columns exist."""
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Normalize dtypes minimally
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
    return df[ANNOTATION_COLUMNS]


def load_annotations(path: str | Path = "app/data/annotations.csv") -> pd.DataFrame:
    """Load annotations CSV if present; otherwise return empty frame with schema."""
    annotations_path = Path(path)
    if annotations_path.exists():
        df = pd.read_csv(annotations_path)
        return ensure_columns(df)
    return ensure_columns(pd.DataFrame(columns=ANNOTATION_COLUMNS))


def reduce_annotations(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest annotation per image_id by timestamp (last-write-wins)."""
    if annotations_df.empty:
        return annotations_df
    df = annotations_df.copy()
    # Parse timestamps; unparseable become NaT and will be treated as older
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Sort by timestamp ascending, then keep the last per image_id
    df = df.sort_values("timestamp")
    latest = df.groupby("image_id", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def apply_annotations(
    predictions_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    id_col: str = "crop_image_id",
    label_col: str = "cropmodel_label",
    set_col: str = "set",
) -> pd.DataFrame:
    """Apply latest annotations to a predictions DataFrame.

    - Override label_col with new_label where present
    - Force set_col to "review" where an override occurs
    """
    if predictions_df is None or len(predictions_df) == 0:
        return predictions_df

    if annotations_df is None or annotations_df.empty:
        return predictions_df

    latest = reduce_annotations(annotations_df)

    left = predictions_df.copy()
    # Ensure id types are comparable
    left[id_col] = left[id_col].astype(str)

    merged = left.merge(
        latest[["image_id", "new_label"]],
        how="left",
        left_on=id_col,
        right_on="image_id",
        validate="m:1",
    )

    # Determine where overrides will occur
    has_override = merged["new_label"].notna() & (merged["new_label"] != "")

    # Override labels
    merged.loc[has_override, label_col] = merged.loc[has_override, "new_label"].astype(str)

    # Ensure set column exists, then set to review on overrides
    if set_col not in merged.columns:
        merged[set_col] = pd.NA
    merged.loc[has_override, set_col] = "review"

    # Drop helper column
    merged = merged.drop(columns=["image_id", "new_label"], errors="ignore")
    return merged


def apply_annotations_to_gdf(
    gdf: "pd.DataFrame",
    annotations_df: pd.DataFrame,
    gdf_image_col: str = "crop_image",
    gdf_label_col: str = "cropmodel_",
    gdf_set_col: str = "set",
) -> "pd.DataFrame":
    """Apply annotations to a GeoDataFrame-like table using crop_image as id.

    Returns a copy with updated label and set for overridden images.
    """
    if gdf is None or len(gdf) == 0:
        return gdf
    if annotations_df is None or annotations_df.empty:
        return gdf

    latest = reduce_annotations(annotations_df)
    left = gdf.copy()
    left[gdf_image_col] = left[gdf_image_col].astype(str)

    merged = left.merge(
        latest[["image_id", "new_label"]],
        how="left",
        left_on=gdf_image_col,
        right_on="image_id",
        validate="m:1",
    )

    has_override = merged["new_label"].notna() & (merged["new_label"] != "")
    merged.loc[has_override, gdf_label_col] = merged.loc[has_override, "new_label"].astype(str)
    if gdf_set_col not in merged.columns:
        merged[gdf_set_col] = pd.NA
    merged.loc[has_override, gdf_set_col] = "review"

    merged = merged.drop(columns=["image_id", "new_label"], errors="ignore")
    return merged


