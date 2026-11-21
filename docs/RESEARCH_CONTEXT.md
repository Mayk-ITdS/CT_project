# Research Context and Problem Definition

## Why fetal CT segmentation is difficult

This project studies fetal ossification centers in volumetric CT data. The technical challenge begins before modeling: scarce access to original scans, heterogeneous source files, very small structures, and variation in anatomy all affect dataset reconstruction and training.

### 1. Clinical context

CT uses ionizing radiation. Its use during pregnancy is limited and requires clinical justification. The medical context makes this a relatively uncommon imaging dataset and helps explain why sample acquisition can be difficult.

The work presented here is a research experiment. It is not a diagnostic device and does not claim clinical validation.

### 2. Data constraints

The source CT volumes and annotations are confidential and cannot be redistributed. The author has confirmed that the figures, screenshots, derived tables and other materials already published in this repository may be used in this portfolio.

The original collection includes raw volumes, mask formats and accompanying metadata that need to be matched and reconstructed before analytical use.

### 3. Computational challenges

| Challenge | Why it matters |
|---|---|
| Tiny targets | The structures occupy only a small fraction of the CT volume, creating extreme foreground/background imbalance. |
| Irregular morphology | Standard geometric assumptions and cropping heuristics are not always reliable. |
| Anatomical variation | Structures change in size and appearance with development. |
| Slice-to-slice changes | A structure may appear, change or disappear in adjacent slices. |
| Subtle contrast | Boundaries between target and surrounding tissue can be difficult to distinguish. |

The research-context infographic prepared for this project illustrates these constraints. Its approximate 0.01% volume-occupancy figure is specific to the analyzed data and is not intended as a universal estimate.

### 4. Consequences for the pipeline

The data preparation work uses reconstructed and aligned CT/mask pairs, spatial and connected-component analysis, region-of-interest extraction and 2.5D training samples. The reported experiment uses a patient-level split, as described by the author, to avoid mixing samples from the same patient across partitions.

See the [main README](../README.md) for the implementation overview, model evaluation, references to existing image assets, and reported metrics.

## Public portfolio versus restricted source data

The original medical source dataset remains private. The publicly approved project figures, code and aggregated results can be displayed to explain the methodology without publishing the restricted volume collection.
