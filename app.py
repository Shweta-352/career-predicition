import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    userdata = [int(x) for x in request.form.values()]
    final = [np.array(userdata)]
    prediction = model.predict(final)
    final_output = np.max(prediction)

    return render_template('result.html', prediction_text="The suitable career option is {}".format(final_output))


if __name__ == "__main__":
    app.run(debug=True)