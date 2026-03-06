import os
from comet_ml import API
from dotenv import load_dotenv
import pandas as pd
import io
from pathlib import Path
import geopandas as gpd
import zipfile

# Load environment variables
load_dotenv()

# Gulf of Mexico flights to include (matches prepare.py / gulf_flights.txt)
GULF_FLIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "gulf_flights.txt"


def _flight_basename(flight_name: str) -> str:
    """Strip first underscore prefix (e.g. 'JPG' from 'JPG_20241219_164400') for gulf_flights.txt matching."""
    if not flight_name:
        return ""
    parts = flight_name.split("_")
    return "_".join(parts[1:]) if len(parts) > 1 else flight_name


def _load_gulf_flights(path: Path = None) -> set:
    """Load flight basenames from gulf_flights.txt (one per line; # and blank ignored). Empty set = no filter."""
    path = path or GULF_FLIGHTS_PATH
    if not path.exists():
        return set()
    basenames = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                basenames.add(line)
    return basenames


def get_all_comet_metrics():
    """Fetch pipeline, detection, and classification metrics from the Comet API.

    - Pipeline: only the latest experiment per flight_name (metrics + final_predictions).
    - Detection and classification: every experiment (metrics only).

    Returns:
        dict with keys 'pipeline', 'detection', 'classification' (DataFrames)
        and 'predictions' (DataFrame).
    """
    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')
    gulf_basenames = _load_gulf_flights()

    print("Fetching experiment list from Comet API...")
    experiments = list(api.get(f"{workspace}/boem"))
    print(f"Found {len(experiments)} total experiments.")

    # First pass: classify each experiment and keep latest pipeline per flight only
    detection_experiments = []   # (exp, flight_name)
    classification_experiments = []
    latest_pipeline_by_flight = {}  # flight_name -> exp (max start_server_timestamp)

    for exp in experiments:
        tags = exp.get_tags()
        is_pipeline = 'pipeline' in tags
        is_detection = 'detection' in tags
        is_classification = 'classification' in tags
        if not (is_pipeline or is_detection or is_classification):
            continue
        if exp.archived or exp.get_state() == 'running':
            continue
        try:
            flight_name = exp.get_parameters_summary("flight_name")["valueCurrent"]
        except Exception:
            flight_name = None

        # Detection and classification are not flight-specific: include all such experiments
        if is_detection:
            detection_experiments.append((exp, flight_name))
        if is_classification:
            classification_experiments.append((exp, flight_name))

        # Gulf filter applies ONLY to pipeline (pipeline is per-flight)
        if not is_pipeline or not flight_name:
            continue
        if gulf_basenames and _flight_basename(flight_name) not in gulf_basenames:
            continue
        exp_ts = getattr(exp, 'start_server_timestamp', None) or 0
        if flight_name not in latest_pipeline_by_flight or exp_ts > getattr(
            latest_pipeline_by_flight[flight_name], 'start_server_timestamp', None
        ) or 0:
            latest_pipeline_by_flight[flight_name] = exp

    print(f"  After filtering: pipeline (latest per flight)=%d, detection=%d, classification=%d" % (
        len(latest_pipeline_by_flight), len(detection_experiments), len(classification_experiments)))

    DETECTION_METRICS = [
        "box_recall", "box_precision", "empty_frame_accuracy",
        "zero_shot_evaluation_box_precision", "zero_shot_evaluation_box_recall",
        "zero_shot_evaluation_empty_frame_accuracy",
    ]
    PIPELINE_METRICS = ["box_recall", "box_precision", "empty_frame_accuracy"]

    pipeline_metrics_data = []
    detection_metrics_data = []
    classification_metrics_data = []
    classification_val_counts = []
    all_predictions = []

    def _normalize_metric_name(name):
        """Normalize for matching: lowercase, spaces/hyphens/slashes to underscores."""
        if not name or not isinstance(name, str):
            return ""
        return name.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")

    def _process_metrics(exp, flight_name, mdf, metrics_to_track):
        if metrics_to_track:
            # Match exactly or via normalized name (Comet UI may show "Zero Shot Evaluation Box Precision")
            canonical = {_normalize_metric_name(m): m for m in metrics_to_track}
            normalized = mdf["metricName"].apply(_normalize_metric_name)
            keep = normalized.isin(canonical)
            mdf = mdf.loc[keep].copy()
            if not mdf.empty:
                mdf["metricName"] = normalized[keep].map(canonical)
        if mdf.empty:
            return None
        mdf = (mdf.sort_values(by='timestamp', ascending=False)
               .groupby('metricName').first().reset_index())
        mdf['timestamp'] = pd.to_datetime(mdf['timestamp'], unit='ms')
        mdf['experiment'] = exp.name
        mdf['experimentKey'] = getattr(exp, 'key', None) or getattr(exp, 'id', None)
        if flight_name:
            mdf['flight_name'] = flight_name
        return mdf

    total_processed = len(latest_pipeline_by_flight) + len(detection_experiments) + len(classification_experiments)
    idx = 0

    # Process latest pipeline per flight (metrics + final_predictions only)
    for flight_name, exp in sorted(latest_pipeline_by_flight.items()):
        idx += 1
        tags = exp.get_tags()
        print(f"  [{idx}/{total_processed}] {exp.name} (flight={flight_name}, type=pipeline [latest])")
        raw_metrics_df = None
        if 'complete' in tags:
            print(f"      Fetching metrics...")
            raw_metrics_df = pd.DataFrame(exp.get_metrics())
        if 'complete' in tags and raw_metrics_df is not None and not raw_metrics_df.empty:
            mdf = _process_metrics(exp, flight_name, raw_metrics_df.copy(), PIPELINE_METRICS)
            if mdf is not None:
                pipeline_metrics_data.append(mdf)
        try:
            print(f"      Fetching final_predictions.csv...")
            final_predictions = exp.get_asset_by_name(
                'final_predictions.csv', asset_type='dataframe')
            final_predictions = pd.read_csv(io.BytesIO(final_predictions))
            final_predictions["flight_name"] = flight_name
            final_predictions["experiment"] = exp.name
            final_predictions["timestamp"] = pd.to_datetime(
                exp.start_server_timestamp, unit='ms')
            all_predictions.append(final_predictions)
        except Exception:
            pass

    # Process all detection experiments (metrics only)
    for exp, flight_name in detection_experiments:
        idx += 1
        print(f"  [{idx}/{total_processed}] {exp.name} (flight={flight_name or 'N/A'}, type=detection)")
        print(f"      Fetching metrics...")
        raw_metrics_df = pd.DataFrame(exp.get_metrics())
        if raw_metrics_df is not None and not raw_metrics_df.empty:
            mdf = _process_metrics(exp, flight_name, raw_metrics_df.copy(), DETECTION_METRICS)
            if mdf is not None:
                detection_metrics_data.append(mdf)

    # Process all classification experiments (metrics + val_annotations)
    for exp, flight_name in classification_experiments:
        idx += 1
        print(f"  [{idx}/{total_processed}] {exp.name} (flight={flight_name or 'N/A'}, type=classification)")
        print(f"      Fetching metrics...")
        raw_metrics_df = pd.DataFrame(exp.get_metrics())
        if raw_metrics_df is not None and not raw_metrics_df.empty:
            mdf = _process_metrics(exp, flight_name, raw_metrics_df.copy(), None)
            if mdf is not None:
                classification_metrics_data.append(mdf)
        try:
            print(f"      Fetching val_annotations.csv...")
            val_asset = exp.get_asset_by_name(
                'val_annotations.csv', asset_type='dataframe')
            val_df = pd.read_csv(io.BytesIO(val_asset))
            label_col = (
                'cropmodel_label' if 'cropmodel_label' in val_df.columns
                else 'label'
            )
            if label_col in val_df.columns:
                counts = (
                    val_df[label_col]
                    .value_counts()
                    .rename_axis('class_name')
                    .reset_index(name='val_support')
                )
                counts['experiment'] = exp.name
                classification_val_counts.append(counts)
        except Exception:
            pass

    # --- Assemble and save results ---
    results = {}

    # Detection
    detection_df = pd.DataFrame()
    if detection_metrics_data:
        detection_df = pd.concat(detection_metrics_data, ignore_index=True)
        detection_df.to_csv("app/data/detection_model_metrics.csv", index=False)
    results['detection'] = detection_df

    # Classification (enriched with val_support)
    classification_df = pd.DataFrame()
    if classification_metrics_data:
        classification_df = pd.concat(classification_metrics_data, ignore_index=True)
        if classification_val_counts:
            val_counts_df = pd.concat(classification_val_counts, ignore_index=True)

            def _class_name_from_metric(name):
                if name and str(name).startswith('Class Accuracy_'):
                    return name[len('Class Accuracy_'):]
                return None

            classification_df['class_name'] = classification_df['metricName'].map(
                _class_name_from_metric)
            classification_df = classification_df.merge(
                val_counts_df, on=['experiment', 'class_name'], how='left')
            totals = (val_counts_df.groupby('experiment')['val_support']
                      .sum().reset_index())
            totals.columns = ['experiment', 'val_support_total']
            classification_df = classification_df.merge(
                totals, on='experiment', how='left')
            classification_df['val_support'] = classification_df['val_support'].fillna(
                classification_df['val_support_total'])
            classification_df = classification_df.drop(
                columns=['val_support_total', 'class_name'])
        classification_df.to_csv(
            "app/data/classification_model_metrics.csv", index=False)
    results['classification'] = classification_df

    # Pipeline
    pipeline_df = pd.DataFrame()
    if pipeline_metrics_data:
        pipeline_df = pd.concat(pipeline_metrics_data, ignore_index=True)
        pipeline_df.to_csv("app/data/metrics.csv", index=False)
    results['pipeline'] = pipeline_df

    # Predictions
    predictions_df = pd.DataFrame()
    if all_predictions:
        predictions_df = pd.concat(all_predictions)
        predictions_df.to_csv("app/data/predictions.csv", index=False)
        latest_dates = predictions_df.groupby(
            'flight_name')['timestamp'].max().reset_index()
        latest_predictions = predictions_df.merge(
            latest_dates, on=['flight_name', 'timestamp'])
        latest_predictions.to_csv(
            "app/data/most_recent_all_flight_predictions.csv", index=False)
    results['predictions'] = predictions_df

    print("Done: detection=%d rows, classification=%d rows, pipeline=%d rows, predictions=%d rows." % (
        len(results['detection']), len(results['classification']),
        len(results['pipeline']), len(results['predictions'])))
    return results

def _has_coords_in_df(df):
    """True if dataframe has lat/lon columns (Lat/Lon or lat/long)."""
    return (
        ('Lat' in df.columns and 'Lon' in df.columns)
        or ('lat' in df.columns and 'long' in df.columns)
    )


def _get_lat_lon_columns(df):
    """Return (lat_col, lon_col) for geometry. Prefer Lat/Lon then lat/long."""
    if 'Lat' in df.columns and 'Lon' in df.columns:
        return 'Lat', 'Lon'
    if 'lat' in df.columns and 'long' in df.columns:
        return 'lat', 'long'
    return None, None


def create_shapefiles(annotations, metadata=None):
    """Create shapefiles for each flight_name.
    Expects annotations to include human_labeled column (from normalize_predictions_scores).
    Shapefile column names truncated to 10 chars: human_labeled -> human_lab

    When annotations already have Lat/Lon (or lat/long), metadata is ignored and no join is done.
    When metadata is provided and annotations lack coords, merges on unique_image to get lat/long.
    """
    annotations = annotations.copy()
    lat_col, lon_col = _get_lat_lon_columns(annotations)

    if lat_col and lon_col:
        # Coords already in predictions (upstream final_predictions.csv); no metadata join
        merged_predictions = annotations
    else:
        # Legacy: join with metadata to get lat/long
        if metadata is None:
            raise ValueError("create_shapefiles: annotations have no Lat/Lon (or lat/long) and metadata path was not provided")
        metadata_df = pd.read_csv(metadata)
        annotations["unique_image"] = annotations["image_path"].apply(lambda x: os.path.splitext(x)[0]).str.split("_").str.join("_")
        metadata_df["unique_image"] = metadata_df["unique_image"].apply(lambda x: x.split("\\")[-1])
        merged_predictions = annotations.merge(metadata_df[["unique_image", "flight_name", "date", "lat", "long"]], on='unique_image')
        lat_col, lon_col = 'lat', 'long'

    # Rename human_labeled to human_lab for shapefile 10-char column limit
    if "human_labeled" in merged_predictions.columns:
        merged_predictions = merged_predictions.rename(columns={"human_labeled": "human_lab"})
    gdf = gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions[lon_col], merged_predictions[lat_col]))
    gdf.crs = "EPSG:4326"
    gdf.to_file("app/data/all_predictions.shp", driver='ESRI Shapefile')

def download_images(experiment_name, save_dir='app/data/images'):
    """Download all images as crops.zip from a Comet experiment and unzip them"""
    save_dir = Path(save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    api = API(api_key=os.getenv('COMET_API_KEY'))
    workspace = os.getenv('COMET_WORKSPACE')

    # Get the experiment object
    experiment = api.get(f"{workspace}/boem", experiment=experiment_name)

    # Find crops.zip asset
    assets = experiment.get_asset_list()
    crops_zip_asset = next((a for a in assets if a['fileName'] == 'crops.zip'), None)
    if crops_zip_asset is None:
        print("No crops.zip found in experiment assets.")
        return

    # Download crops.zip
    zip_data = experiment.get_asset(crops_zip_asset['assetId'])
    zip_path = save_dir / f'{experiment.name}.zip'
    with open(zip_path, 'wb') as f:
        f.write(zip_data)

    # Unzip crops.zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    zip_path.unlink()  # Remove the zip file after extraction


def download_flythrough_videos(flight_to_experiment, save_dir="app/data/videos"):
    """Download the flythrough video per flight from Comet (asset named {flight_name}_flythrough.mp4).
    Uses the experiment already chosen per flight (e.g. from latest_predictions); no experiment list fetch.
    flight_to_experiment: dict mapping flight_name -> experiment name.
    """
    if not flight_to_experiment:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    api = API(api_key=os.getenv("COMET_API_KEY"))
    workspace = os.getenv("COMET_WORKSPACE")

    for flight_name, experiment_name in flight_to_experiment.items():
        asset_name = f"{flight_name}_flythrough.mp4"
        out_path = save_dir / asset_name
        try:
            experiment = api.get(f"{workspace}/boem", experiment=experiment_name)
            assets = experiment.get_asset_list()
            asset = next((a for a in assets if a.get("fileName") == asset_name), None)
            if asset is None:
                print(f"  No asset {asset_name} in {experiment_name}, skipping.")
                continue
            print(f"  Downloading {asset_name} from {experiment_name}...")
            data = experiment.get_asset(asset["assetId"])
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"    Saved to {out_path}")
        except Exception as e:
            print(f"    Failed to download {flight_name}_flythrough.mp4: {e}")


def download_flight_reports(flight_to_experiment, save_dir="app/data/reports"):
    """Download the latest report assets per flight from Comet (folder containing transect_map.html, report.pdf, etc.).
    Uses the same experiment mapping as flythrough videos. Saves each flight's report folder under save_dir/flight_name/.
    """
    if not flight_to_experiment:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    api = API(api_key=os.getenv("COMET_API_KEY"))
    workspace = os.getenv("COMET_WORKSPACE")

    for flight_name, experiment_name in flight_to_experiment.items():
        out_dir = save_dir / flight_name
        try:
            experiment = api.get(f"{workspace}/boem", experiment=experiment_name)
            assets = experiment.get_asset_list()
            # Find report folder: asset with transect_map.html (path may be "folder/transect_map.html" or "others/folder/transect_map.html")
            report_prefix = None
            for a in assets:
                fname = (a.get("fileName") or a.get("filePath") or "").strip()
                if "transect_map.html" in fname:
                    # Use the directory containing transect_map.html as the report folder
                    if "/" in fname:
                        report_prefix = fname.rsplit("/", 1)[0] + "/"
                    else:
                        report_prefix = ""
                    break
            if not report_prefix:
                print(f"  No report folder (transect_map.html) in {experiment_name}, skipping.")
                continue
            # Download all assets under that prefix
            to_download = [
                a for a in assets
                if (a.get("fileName") or a.get("filePath") or "").startswith(report_prefix)
            ]
            if not to_download:
                print(f"  No assets under report folder in {experiment_name}, skipping.")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for a in to_download:
                fname = (a.get("fileName") or a.get("filePath") or "").strip()
                if "/" in fname:
                    rel_path = fname[len(report_prefix):] or fname.split("/")[-1]
                else:
                    rel_path = fname
                if not rel_path:
                    continue
                out_path = out_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    data = experiment.get_asset(a["assetId"])
                    with open(out_path, "wb") as f:
                        f.write(data)
                except Exception as e:
                    print(f"    Failed to download {rel_path}: {e}")
            print(f"  Downloaded report for {flight_name} to {out_dir} ({len(to_download)} files)")
        except Exception as e:
            print(f"  Failed to download report for {flight_name}: {e}")
