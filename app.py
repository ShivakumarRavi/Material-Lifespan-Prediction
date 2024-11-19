from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            ComponentType=request.form.get("componenttype"),
            StructureType=request.form.get("structuretype"),
            CoolRate=request.form.get("coolrate"),
            QuenchDuration=request.form.get("quenchduration"),
            ForgeDuration=request.form.get("forgeduration"),
            HeatProcessTime=request.form.get("heatprocesstime"),
            NickelComposition=request.form.get("nickelcomposition"),
            IronComposition=request.form.get("ironcomposition"),
            CobaltComposition=request.form.get("cobaltcomposition"),
            ChromiumComposition=request.form.get("chromiumcomposition"),
            MinorDefects=request.form.get("minordefects"),
            MajorDefects=request.form.get("majordefects"),
            EdgeDefects=request.form.get("edgedefects"),
            InitialPosition=request.form.get("initialposition"),
            FormationMethod=request.form.get("formationmethod"),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")
