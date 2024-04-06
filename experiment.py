# import subprocess
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def pythoncode(x, y):
    # Read the dataset
    df = pd.read_csv("PBL.csv")

    # Features and target columns
    X = df[["CET_Cutoff ", "Category"]]
    y_college = df["College_Name"]
    y_branch = df["Branch"]

    # Train-test split
    (
        X_train,
        X_test,
        y_college_train,
        y_college_test,
        y_branch_train,
        y_branch_test,
    ) = train_test_split(X, y_college, y_branch, test_size=0.2, random_state=42)

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Category"])],
        remainder="passthrough",
    )

    # Create the pipeline
    model_college = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model_branch = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Train the model
    model_college.fit(X_train, y_college_train)
    model_branch.fit(X_train, y_branch_train)

    # Predictions
    cet_cutoff_input = float(x)
    category_input = str(y)

    input_data = pd.DataFrame(
        {"CET_Cutoff ": [cet_cutoff_input], "Category": [category_input]}
    )

    college_prediction = model_college.predict(input_data)
    branch_prediction = model_branch.predict(input_data)
    return {
        "college_prediction": college_prediction[0],
        "branch_prediction": branch_prediction[0],
    }


@app.route("/api/add", methods=["POST"])
def add():
    try:
        data = request.get_json()
        result = pythoncode(int(data["percentile"]), int(data["category"]))
        return jsonify({"result": result, "success": True})
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
