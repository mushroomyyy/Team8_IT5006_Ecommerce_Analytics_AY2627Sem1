# Phase 2 Model Development Handoff

Recorded on: 20 September 2026  
Related notebook: [`model_dev.ipynb`](./model_dev.ipynb)

## Project reference

- [IT5006 Project Description — live website](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html)

## Current notebook scope

The notebook has been pushed to `main` and provides a basic structure for end-to-end model development:

1. Load the data.
2. Perform basic feature engineering.
3. Combine all datasets.
4. Create the train/test split.
5. Train the models.
6. Evaluate model performance.

## Decisions and implementation notes

- **Train/test split:** A proportional split is currently used. A sequential month-based split was considered, but not implemented because late-delivery rates fluctuate considerably from month to month.
- **Model training:** The initial Random Forest (RF) and XGBoost (XGB) implementations were adapted from the recorded lecture.
- **Model evaluation:** The notebook includes the standard evaluation metrics from the lecture.
- **Coverage summary:** An additional "Coverage" summary table was included. This is a classification-model performance view the contributor's team commonly uses when explaining results to stakeholders.
- **Prediction granularity:** The notebook was subsequently updated from monthly prediction to daily prediction because monthly prediction was considered unsuitable.

## Intended use

Treat this notebook as the starting structure for Phase 2 model development. The modeling choices, validation strategy, feature engineering, and evaluation approach can now be reviewed and refined by the group.
