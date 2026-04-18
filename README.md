# DOPAVALUE Analysis Pipeline

A pipeline for quantifying fluorescently labelled objects in brain section images stored in [OMERO](https://www.openmicroscopy.org/omero/). Images are cropped to anatomical ROIs, classified with [ilastik](https://www.ilastik.org/), segmented, and aggregated into measurement tables.

## Pipeline overview

```
[OMERO dataset with annotated images]
          |
          v
1. omero_download_training_rois   →  .npy files for ilastik training
          |
          v
   [Train ilastik classifier]
          |
          v
2. omero_export_all_rois          →  OMERO dataset: <name>_rois
          |                           (MIP + AIP images per ROI)
          v
3. omero_run_ilastik              →  OMERO dataset: <name>_ilastik_output
          |                           (probability maps, mask ROIs, point ROIs,
          |                            per-object CSV annotations, summary table)
          v
4. omero_filter_objects           →  filtered tables attached to dataset
                                      (Aggregated_measurements_filtered)
```

## Prerequisites

### Software
- Python 3.9 (see `binder/environment.yml`)
- [ilastik](https://www.ilastik.org/) ≥ 1.4.0 installed locally
- Access to an OMERO server

### Python dependencies

Install via conda using the provided environment file:

```bash
conda env create -f binder/environment.yml
conda activate dopavalue
```

Additional pip dependencies:
```bash
pip install omero-py omero2pandas omero-rois scikit-image pandas numpy
```

## Image naming convention

Images must be named with underscore-separated metadata fields:

```
{mouse_id}_{AP}_{section_nr}_{date}_{magnification}_{markers}
```

Example: `M001_-2.0_03_20240115_20x_EGFP-DsRed`

This convention is used to populate the metadata columns of the output measurement tables.

## Scripts

### 1. `omero_download_training_rois.py`

Downloads intensity data from ROIs labelled `training` within a tagged subset of images, and saves them as `.npy` files for use in ilastik model training.

**Configure before running:**
| Variable | Description |
|---|---|
| `OUTPUT_DIR` | Local directory to save `.npy` training files |
| `HOST` / `PORT` | OMERO server address |
| `ROI_COMMENTS` | ROI label(s) to download (default: `['training']`) |
| `channel` | Channel index to extract |

**Prompts at runtime:** username, password, OMERO group, dataset ID, tag text (default: `training_set`).

---

### 2. `omero_export_all_rois.py`

For each image in a dataset, extracts the pixel data inside each matching ROI and uploads it back to OMERO as two new images:
- `<image_name>_<roi_label>_MIP` — Maximum Intensity Projection
- `<image_name>_<roi_label>_AIP` — Average Intensity Projection

A new dataset named `<source_dataset>_rois` is created in the same project.

**Configure before running:**
| Variable | Description |
|---|---|
| `HOST` / `PORT` | OMERO server address |
| `ROI_COMMENTS` | Tuple of ROI labels to export (e.g. `("lprl", "rprl", "lil", "ril")`). Set to `None` to export all ROIs. |
| `C_RANGE` | Channel range to export (`None` = all channels) |

**Accepts command-line arguments** (in order): `username password group dataset_id`. Falls back to interactive prompts when not provided.

```bash
python omero_export_all_rois.py <username> <password> <group> <dataset_id>
```

---

### 3. `omero_run_ilastik.py`

The main analysis script. For each MIP/AIP image pair in the `_rois` dataset:

1. Saves the MIP image locally as a temporary `.npy` file.
2. Runs ilastik headless to generate a pixel-probability map (saved as `.npy` and `.tiff`).
3. Segments objects and background from the probability channels using hysteresis thresholding.
4. Uploads results back to OMERO:
   - **Probability image** in the `<dataset>_ilastik_output` dataset.
   - **Mask ROIs** overlaid on the MIP image.
   - **Point ROIs** at object centroids — red if `area > SIZE_LIMIT`, green otherwise.
   - **CSV file annotations** (`ch<n>_object_df.csv`, `ch<n>_bg_df.csv`) attached to the AIP image.
5. Computes per-channel aggregated statistics and writes an `Aggregated_measurements` OMERO table attached to the dataset.

**Configure before running:**
| Variable | Description |
|---|---|
| `HOST` / `PORT` | OMERO server address |
| `TEMP_DIR` | Local directory for temporary files |
| `ILASTIK_PATH` | Path to the ilastik `run_ilastik.sh` script |
| `PROJECT_PATH` | Path to the trained ilastik `.ilp` project file |
| `object_ch_match` | `(object_channel, probability_channel)` pairs for objects |
| `ch_bg_match` | `(object_channel, probability_channel)` pairs for background |
| `ch_names` | Channel names in order (e.g. `["DAPI", "EGFP", "DsRed", "Cy5"]`) |
| `segmentation_thr` | Hysteresis threshold per channel (0–255) |
| `upper_correction_factors` | High threshold multiplier per channel |
| `lower_correction_factors` | Low threshold multiplier per channel |
| `SIZE_LIMIT` | Area threshold (pixels) for colour-coding detected objects |

**Prompts at runtime:** username, password, server, port, group, dataset ID.

**Output columns per channel** (in the aggregated table):

| Column | Description |
|---|---|
| `roi_intensity_<ch>` | Total intensity summed over the ROI |
| `object_count_<ch>` | Number of detected objects |
| `mean_area_<ch>` | Mean object area (pixels) |
| `median_area_<ch>` | Median object area (pixels) |
| `sum_area_<ch>` | Total object area (pixels) |
| `sum_intensity_<ch>` | Summed integrated intensity of all objects |
| `mean_intensity_<ch>` | Mean intensity across all object pixels |
| `sum_area_bg_<ch>` | Total background area |
| `sum_intensity_bg_<ch>` | Summed background intensity |
| `mean_intensity_bg_<ch>` | Mean background intensity |

---

### 4. `omero_filter_objects.py`

Applies manual curation decisions to the ilastik output. After reviewing the point ROIs overlaid on MIP images in OMERO, users can delete unwanted point ROIs directly in the OMERO client. This script then:

1. Reads the surviving point ROI labels from OMERO.
2. Re-filters the per-object CSV dataframes to keep only those labels.
3. Objects removed from the object set are moved into the background dataframe.
4. Uploads filtered tables (`ch<n>_object_df_filtered`, `ch<n>_bg_df_filtered`) back to OMERO.
5. Recomputes aggregated statistics and attaches an `Aggregated_measurements_filtered` table to the dataset.

**Configure before running:**
| Variable | Description |
|---|---|
| `HOST` / `PORT` | OMERO server address |
| `ch_names` | Must match the channel names used in step 3 |

**Prompts at runtime:** username, password, server, port, group, dataset ID (the `_rois` dataset from step 2).

---

## Supporting module: `omero_toolbox.py`

A utility library used by all scripts above. Key functionality:

- **Connection management**: `open_connection`, `close_connection`
- **Data retrieval**: `get_intensities`, `get_shape_intensities` (rectangle and polygon ROIs, with optional polygon masking)
- **Image creation**: `create_image_from_numpy_array` (handles tiled upload for large images)
- **ROI creation**: `create_shape_point`, `create_shape_mask`, `create_shape_polygon`, `create_roi`
- **Annotation utilities**: `create_annotation_table`, `create_annotation_file_local`, `create_annotation_map`, `link_annotation`
- **Dataset/project management**: `create_dataset`, `create_project`, `link_dataset_to_project`

## Notes

- All array dimensions follow the OMERO/ilastik convention: **Z, C, T, Y, X**.
- The ilastik model must be trained separately using the `.npy` files produced by `omero_download_training_rois.py`. The probability output is expected to have object and background as distinct channels (configured via `object_ch_match` / `ch_bg_match`).
- Temporary files in `TEMP_DIR` are not cleaned up automatically.
