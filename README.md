# Malaysian Traffic Sign Detection and Recognition

This submission contains a Flask application that detects Malaysian traffic signs
and recognises each detected sign class.

## Project layout

```text
Application/                 Flask application, templates, static assets, and local history
Model Development/scripts/   Dataset-preparation, training, and evaluation scripts
Model Weights/               Trained runtime models
Configuration/               Dependencies and class-label configuration
```

The runtime application needs both trained model files:

- `Model Weights/detection_best.pt` locates traffic signs.
- `Model Weights/recognition_best.pt` classifies each detected sign crop.

`Configuration/class_labels.csv` is a small label mapping used to show readable
sign names. Training datasets are intentionally excluded from this source-code
submission; they are only required when developing or retraining models.

## Run the application

From the project root:

```powershell
py -m pip install -r "Configuration/requirements.txt"
py "Application/app.py"
```

Then open `http://127.0.0.1:5000`.

## Model development

All development scripts are in `Model Development/scripts/`. To retrain a model,
first place or regenerate the required training dataset outside this submission,
then provide its path explicitly. For example:

```powershell
py "Model Development/scripts/train_yolo_detection.py" --data path\to\data.yaml
py "Model Development/scripts/train_recognition.py" --data path\to\recognition_dataset