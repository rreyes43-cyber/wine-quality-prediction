# Wine Quality Prediction System

A Streamlit classroom prototype that classifies red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT** using a Random Forest model.

## Dataset
UCI Machine Learning Repository – Wine Quality (red wine):
https://archive.ics.uci.edu/dataset/186/wine+quality

The project downloads `winequality-red.csv` automatically when `train_model.py` runs. The original CSV is semicolon-separated.

## Prediction rule
- `quality >= 6` → GOOD QUALITY (1)
- `quality < 6` → NEEDS IMPROVEMENT (0)

The original `quality` column is the target and is excluded from model input features to prevent target leakage.

## Files
- `train_model.py` – downloads data, removes duplicate rows for training, trains and evaluates the Random Forest pipeline.
- `app.py` – Streamlit dashboard, CSV upload, predictions, and result download.
- `requirements.txt` – Python dependencies.
- `wine_quality.csv`, `wine_model.pkl`, `sample_wine_input.csv`, `evaluation_metrics.txt` – generated when training runs.

## Run locally
Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). Download the sample input CSV from the app, then upload it to test predictions.

## Deploy on Streamlit Community Cloud
1. Push `train_model.py`, `app.py`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository, branch (`main`), and main file (`app.py`).
4. Deploy. The app will install `requirements.txt`.
5. Because the model and dataset are generated locally by the training script, run `python train_model.py` before deployment and commit `wine_model.pkl` and `wine_quality.csv` to the repository, OR modify deployment to train/download artifacts at startup. For this classroom activity, committing the generated model and CSV is the simplest approach if allowed by the instructor.
6. Copy the public app URL after deployment and test it in a private/incognito browser window.

## Limitations
The model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. Predictions are only preliminary screening results and must not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## Dataset citation
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T
