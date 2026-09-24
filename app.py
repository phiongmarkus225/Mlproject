"""Flask web application for the Student Performance ML project.

This is the entry point of the web app. It serves:
- GET  /            -> the prediction form (frontend)
- POST /predictdata -> receives form input, runs the predict pipeline,
                       and shows the predicted math score
- GET  /health      -> health check used by Docker/CI
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from flask import Flask, render_template, request

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the prediction form."""
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Handle the form submission and return the prediction."""
    if request.method == "GET":
        return render_template("index.html")

    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score")),
        )

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(
            data.get_data_as_data_frame()
        )

        logging.info(f"Prediction returned: {results:.2f}")

        return render_template(
            "index.html",
            results=round(results, 2),
        )

    except Exception as e:
        raise CustomException(e, sys)


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)