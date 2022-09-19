from flask import Flask,render_template,request
from artifacts.utils import ifp

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.form
    ifp_obj = ifp(data)
    result = ifp_obj.predict()
    return render_template("index.html",pred=result)

if __name__ == "__main__":
    app.run(debug=True)
