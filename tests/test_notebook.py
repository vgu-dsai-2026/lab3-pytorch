from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

# -----------------------------------------------------------------------------
# MODULE BOOTSTRAP
# -----------------------------------------------------------------------------

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebook.py"
SPEC = importlib.util.spec_from_file_location("student_notebook", NOTEBOOK_PATH)
NOTEBOOK_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(NOTEBOOK_MODULE)

# Public API imported from student notebook
list_image_paths_for_group   = NOTEBOOK_MODULE.list_image_paths_for_group
inspect_image_file           = NOTEBOOK_MODULE.inspect_image_file
build_metadata_from_folders  = NOTEBOOK_MODULE.build_metadata_from_folders
load_metadata_table          = NOTEBOOK_MODULE.load_metadata_table
summarize_metadata           = NOTEBOOK_MODULE.summarize_metadata
build_label_split_table      = NOTEBOOK_MODULE.build_label_split_table
audit_metadata               = NOTEBOOK_MODULE.audit_metadata
add_analysis_columns         = NOTEBOOK_MODULE.add_analysis_columns
build_split_characteristics_table = NOTEBOOK_MODULE.build_split_characteristics_table
sample_balanced_by_split_and_label = NOTEBOOK_MODULE.sample_balanced_by_split_and_label

# Safe project root (works in scripts + notebooks)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_ROOT              = PROJECT_ROOT / "data"
GENERATED_METADATA_PATH = PROJECT_ROOT / "artifacts" / f"lab2_faces_metadata.csv"
SEED                   = NOTEBOOK_MODULE.SEED


# -----------------------------------------------------------------------------
# Question 1: list_image_paths_for_group  /  inspect_image_file
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "label"),
    [
        pytest.param("train", "cat", id="train-cat-returns-nonempty-paths"),
        pytest.param("test",  "dog", id="test-dog-returns-nonempty-paths"),
    ],
)
def test_list_image_paths_for_group_nonempty(split: str, label: str) -> None:
    paths = list_image_paths_for_group(DATA_ROOT, split, label)

    print(paths)

    assert isinstance(paths, list), "Result must be a list"
    assert len(paths) > 0, f"Expected at least one image for split={split}, label={label}"
    assert all(isinstance(p, Path) for p in paths), "Every item must be a Path"
    assert all(p.exists() for p in paths), "Every path must exist on disk"
    assert all(
        p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in paths
    ), "Every path must point to an image file"

@pytest.mark.parametrize(
    ("filepath", "expected_width", "expected_height", "expected_mean"),
    [
        ("test/cat/cat_0009.jpg", 64, 64, 0.657925),
        ("test/cat/cat_0010.jpg", 64, 64, 0.454074),
        ("test/dog/dog_0009.jpg", 64, 64, 0.274566),
        ("test/dog/dog_0010.jpg", 64, 64, 0.537693),
    ],
)
def test_inspect_image_file_exact_values(filepath, expected_width, expected_height, expected_mean):
    path = DATA_ROOT / filepath

    assert path.exists(), f"File does not exist: {path}"

    width, height, mean_intensity = inspect_image_file(path)

    # Exact checks for integers
    assert width == expected_width, f"Width mismatch for {filepath}"
    assert height == expected_height, f"Height mismatch for {filepath}"

    # Approximate check for float
    assert mean_intensity == pytest.approx(expected_mean, rel=1e-3), (
        f"Mean intensity mismatch for {filepath}: "
        f"expected {expected_mean}, got {mean_intensity}"
    )

# -----------------------------------------------------------------------------
# Question 2: load_metadata_table
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def metadata_path() -> Path:
    GENERATED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # build_metadata_from_folders should create the CSV file
    folder_df = build_metadata_from_folders(DATA_ROOT)
    folder_df.to_csv(GENERATED_METADATA_PATH, index=False)


    assert GENERATED_METADATA_PATH.exists(), (
        f"Metadata file was not created: {GENERATED_METADATA_PATH}"
    )
    return GENERATED_METADATA_PATH

def test_build_and_load_metadata(metadata_path: Path) -> None:
    df = load_metadata_table(metadata_path)

    assert isinstance(df, pd.DataFrame)
    assert metadata_path.exists()

    required_columns = {
        "filepath", "label", "split",
        "width", "height", "mean_intensity"
    }
    assert required_columns.issubset(df.columns)


@pytest.mark.parametrize(
    ("column", "expected_dtype_kind"),
    [
        ("width", "i"),
        ("mean_intensity", "f"),
    ],
)
def test_load_metadata_table_dtypes(metadata_path: Path, column: str, expected_dtype_kind: str):
    df = load_metadata_table(metadata_path)

    assert df[column].dtype.kind == expected_dtype_kind


# -----------------------------------------------------------------------------
# Question 3: summarize_metadata
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key",),
    [
        pytest.param("rows",         id="summary-contains-rows-key"),
        pytest.param("columns",      id="summary-contains-columns-key"),
        pytest.param("class_counts", id="summary-contains-class-counts-key"),
        pytest.param("split_counts", id="summary-contains-split-counts-key"),
    ],
)
def test_summarize_metadata_keys(key: str) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    summary = summarize_metadata(df)

    assert isinstance(summary, dict), "Result must be a dict"
    assert key in summary, f"Key '{key}' is missing from the summary dict"


@pytest.mark.parametrize(
    ("expected_labels", "expected_splits"),
    [
        pytest.param(
            {"cat", "dog"},
            {"train", "val", "test"},
            id="class-counts-and-split-counts-cover-all-groups",
        )
    ],
)
def test_summarize_metadata_values(
    expected_labels: set[str], expected_splits: set[str]
) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    summary = summarize_metadata(df)

    assert summary["rows"] == len(df), (
        f"summary['rows'] is {summary['rows']} but DataFrame has {len(df)} rows"
    )
    assert set(summary["class_counts"].index) == expected_labels, (
        f"class_counts index {set(summary['class_counts'].index)} != {expected_labels}"
    )
    assert set(summary["split_counts"].index) == expected_splits, (
        f"split_counts index {set(summary['split_counts'].index)} != {expected_splits}"
    )

def test_summarize_metadata_exact_values():
    df = load_metadata_table(GENERATED_METADATA_PATH)
    summary = summarize_metadata(df)

    # Class counts
    expected_class_counts = {"cat": 10, "dog": 10}
    actual_class_counts = summary["class_counts"].to_dict()

    assert actual_class_counts == expected_class_counts, (
        f"Expected class counts {expected_class_counts}, got {actual_class_counts}"
    )

    # Split counts (total)
    expected_split_counts = {"train": 12, "val": 4, "test": 4}
    actual_split_counts = summary["split_counts"].to_dict()

    assert actual_split_counts == expected_split_counts, (
        f"Expected split counts {expected_split_counts}, got {actual_split_counts}"
    )


# -----------------------------------------------------------------------------
# Question 4: build_label_split_table
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected_index", "expected_columns"),
    [
        pytest.param(
            {"cat", "dog"},
            {"train", "val", "test"},
            id="labels-are-index-splits-are-columns",
        )
    ],
)
def test_build_label_split_table_structure(
    expected_index: set[str], expected_columns: set[str]
) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    table = build_label_split_table(df)

    assert isinstance(table, pd.DataFrame), "Result must be a DataFrame"
    assert expected_index.issubset(set(table.index)), (
        f"Table index {set(table.index)} must include {expected_index}"
    )
    assert expected_columns.issubset(set(table.columns)), (
        f"Table columns {set(table.columns)} must include {expected_columns}"
    )


@pytest.mark.parametrize(
    ("label", "split"),
    [
        pytest.param("cat", "train", id="cat-train-cell-matches-raw-count"),
        pytest.param("dog", "test",  id="dog-test-cell-matches-raw-count"),
    ],
)
def test_build_label_split_table_counts(label: str, split: str) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    table = build_label_split_table(df)

    expected = int(((df["label"] == label) & (df["split"] == split)).sum())
    actual   = int(table.loc[label, split])
    assert actual == expected, (
        f"table.loc['{label}', '{split}'] == {actual}, expected {expected}"
    )


# -----------------------------------------------------------------------------
# Question 5: audit_metadata
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key",),
    [
        pytest.param("missing_values",     id="audit-contains-missing-values-key"),
        pytest.param("duplicate_filepaths", id="audit-contains-duplicate-filepaths-key"),
        pytest.param("bad_labels",          id="audit-contains-bad-labels-key"),
        pytest.param("non_positive_sizes",  id="audit-contains-non-positive-sizes-key"),
    ],
)
def test_audit_metadata_keys(key: str) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    report = audit_metadata(df)

    assert isinstance(report, dict), "Result must be a dict"
    assert key in report, f"Key '{key}' is missing from the audit report"


@pytest.mark.parametrize(
    ("injected_label", "should_flag"),
    [
        pytest.param("bird",  True,  id="injected-bad-label-is-flagged"),
        pytest.param("cat",   False, id="clean-dataframe-has-no-bad-labels"),
    ],
)
def test_audit_metadata_bad_labels(injected_label: str, should_flag: bool) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)

    if should_flag:
        dirty = df.copy()
        dirty.loc[0, "label"] = injected_label
        report = audit_metadata(dirty)
        assert injected_label in report["bad_labels"], (
            f"'{injected_label}' should appear in bad_labels but got: {report['bad_labels']}"
        )
    else:
        report = audit_metadata(df)
        assert report["bad_labels"] == [], (
            f"Clean DataFrame should have no bad_labels, got: {report['bad_labels']}"
        )

def test_audit_metadata_missing_values():
    df = load_metadata_table(GENERATED_METADATA_PATH)

    dirty = df.copy()
    dirty.loc[0, "width"] = np.nan

    report = audit_metadata(dirty)

    assert report["missing_values"]["width"] > 0

def test_audit_metadata_duplicate_filepaths():
    df = load_metadata_table(GENERATED_METADATA_PATH)

    dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = audit_metadata(dirty)

    assert report["duplicate_filepaths"] > 0, (
        f"Expected duplicate_filepaths > 0, got {report['duplicate_filepaths']}"
    )

def test_audit_metadata_non_positive_sizes():
    df = load_metadata_table(GENERATED_METADATA_PATH)

    dirty = df.copy()
    dirty.loc[0, "width"] = 0   # invalid
    dirty.loc[1, "height"] = -1 # invalid

    report = audit_metadata(dirty)

    assert report["non_positive_sizes"] > 0, (
        f"Expected non_positive_sizes > 0, got {report['non_positive_sizes']}"
    )

# -----------------------------------------------------------------------------
# Question 6: add_analysis_columns
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column",),
    [
        pytest.param("pixel_count",     id="pixel-count-column-added"),
        pytest.param("aspect_ratio",    id="aspect-ratio-column-added"),
        pytest.param("brightness_band", id="brightness-band-column-added"),
        pytest.param("size_bucket",     id="size-bucket-column-added"),
    ],
)
def test_add_analysis_columns_present(column: str) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    result = add_analysis_columns(df)

    assert column in result.columns, f"Column '{column}' is missing from the result"


def test_add_analysis_columns_exact_values():
    df = pd.DataFrame({
        "width": [10, 20],
        "height": [5, 4],
        "mean_intensity": [0.2, 0.8],
        "label": ["cat", "dog"],
        "split": ["train", "val"],
        "filepath": ["a.jpg", "b.jpg"],
    })

    result = add_analysis_columns(df)

    # pixel_count = width * height
    expected_pixel = pd.Series([50, 80], name="pixel_count")
    pd.testing.assert_series_equal(
        result["pixel_count"].reset_index(drop=True),
        expected_pixel,
        check_dtype=False,
    )

    # aspect_ratio = width / height
    expected_ratio = pd.Series([2.0, 5.0], name="aspect_ratio")
    pd.testing.assert_series_equal(
        result["aspect_ratio"].reset_index(drop=True),
        expected_ratio,
        check_dtype=False,
    )


# ---------------------------------------------------------------------
# brightness_band test (controlled quantiles)
# ---------------------------------------------------------------------

def test_add_analysis_columns_brightness_band():
    df = pd.DataFrame({
        "width": [10] * 8,
        "height": [10] * 8,
        "mean_intensity": [
            0.1, 0.2,   # dark
            0.3, 0.4,   # dim
            0.5, 0.6,   # bright
            0.7, 0.8    # very_bright
        ],
        "label": ["cat"] * 8,
        "split": ["train"] * 8,
        "filepath": [f"img_{i}.jpg" for i in range(8)],
    })

    result = add_analysis_columns(df)

    expected_labels = [
        "darkest", "darkest",
        "dim", "dim",
        "bright", "bright",
        "brightest", "brightest",
    ]

    actual_labels = result["brightness_band"].astype(str).tolist()

    assert actual_labels == expected_labels, (
        f"Expected {expected_labels}, got {actual_labels}"
    )


# ---------------------------------------------------------------------
# size_bucket test (relative behavior, not implementation-specific)
# ---------------------------------------------------------------------

def test_add_analysis_columns_size_bucket():
    reference = 64 * 64  # expected reference size

    df = pd.DataFrame({
        "width":  [32, 64, 128],
        "height": [32, 64, 128],
        "mean_intensity": [0.2, 0.5, 0.8],
        "label": ["cat", "cat", "dog"],
        "split": ["train", "train", "train"],
        "filepath": ["small.jpg", "medium.jpg", "large.jpg"],
    })

    result = add_analysis_columns(df)

    pixel_counts = result["pixel_count"]
    size_bucket  = result["size_bucket"].astype(str)

    # Check relative correctness instead of exact implementation
    assert size_bucket.iloc[0] == "small", "Expected small for smaller-than-reference image"
    assert size_bucket.iloc[1] == "medium", "Expected medium for reference-sized image"
    assert size_bucket.iloc[2] == "large", "Expected large for larger-than-reference image"


# -----------------------------------------------------------------------------
# Question 7: build_split_characteristics_table
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected_index", "expected_columns"),
    [
        pytest.param(
            {"train", "val", "test"},
            {"avg_width", "avg_height", "avg_pixel_count", "avg_mean_intensity"},
            id="splits-are-index-numeric-summary-columns",
        )
    ],
)
def test_build_split_characteristics_table_structure(
    expected_index: set[str], expected_columns: set[str]
) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    analysis = add_analysis_columns(df)
    table = build_split_characteristics_table(analysis)

    assert isinstance(table, pd.DataFrame), "Result must be a DataFrame"
    assert expected_index.issubset(set(table.index)), (
        f"Table index {set(table.index)} must include {expected_index}"
    )
    assert expected_columns.issubset(set(table.columns)), (
        f"Table columns {set(table.columns)} must include {expected_columns}"
    )


@pytest.mark.parametrize(
    ("split", "column"),
    [
        pytest.param("train", "avg_width", id="train-width-mean-matches-groupby-result"),
        pytest.param("val", "avg_mean_intensity", id="val-brightness-mean-matches-groupby-result"),
    ],
)
def test_build_split_characteristics_table_values(split: str, column: str) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    analysis = add_analysis_columns(df)
    table = build_split_characteristics_table(analysis)

    expected = (
        analysis.groupby("split")[["width", "height", "pixel_count", "mean_intensity"]]
        .mean()
        .rename(
            columns={
                "width": "avg_width",
                "height": "avg_height",
                "pixel_count": "avg_pixel_count",
                "mean_intensity": "avg_mean_intensity",
            }
        )
    )
    actual = float(table.loc[split, column])
    assert actual == pytest.approx(float(expected.loc[split, column]), rel=1e-6), (
        f"table.loc['{split}', '{column}'] == {actual}, expected {expected.loc[split, column]}"
    )


# -----------------------------------------------------------------------------
# Question 8: sample_balanced_by_split_and_label
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_per_group", "seed"),
    [
        pytest.param(1, SEED,   id="n=1-shape-respects-small-groups"),
        pytest.param(5, SEED, id="n=5-shape-respects-small-groups"),
    ],
)
def test_sample_balanced_shape(n_per_group: int, seed: int) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    analysis = add_analysis_columns(df)

    grouped = analysis.groupby(["split", "label"])
    expected_rows = sum(min(n_per_group, len(group)) for _, group in grouped)

    sampled = sample_balanced_by_split_and_label(
        analysis, n_per_group=n_per_group, seed=seed
    )

    assert sampled.shape[0] == expected_rows, (
        f"Expected {expected_rows} rows, got {sampled.shape[0]}"
    )


@pytest.mark.parametrize(
    ("n_per_group", "seed"),
    [
        pytest.param(1, SEED,   id="n=1-each-group-respects-limit"),
        pytest.param(5, SEED, id="n=5-each-group-respects-limit"),
    ],
)
def test_sample_balanced_per_group_counts(n_per_group: int, seed: int) -> None:
    df = load_metadata_table(GENERATED_METADATA_PATH)
    analysis = add_analysis_columns(df)

    sampled = sample_balanced_by_split_and_label(
        analysis, n_per_group=n_per_group, seed=seed
    )

    original_sizes = analysis.groupby(["split", "label"]).size()
    sampled_sizes  = sampled.groupby(["split", "label"]).size()

    for key in original_sizes.index:
        expected = min(n_per_group, original_sizes[key])
        actual   = sampled_sizes.get(key, 0)

        assert actual == expected, (
            f"{key}: expected {expected}, got {actual}"
        )