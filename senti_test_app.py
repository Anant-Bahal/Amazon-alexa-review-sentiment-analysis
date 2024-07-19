from flask import Flask, request,  render_template
import re
from io import BytesIO


from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

# Create flask app
flask_app = Flask(__name__)
predictor = pickle.load(open(r"model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"scaler.pkl", "rb"))
cv = pickle.load(open(r"countVectorizer.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("indexi.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    text_input = request.form.get('textarea') 
 
       # Print the text in terminal for verification 
    
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    if y_predictions == 1:
        ans="positive"
    else:
        ans="negative"
    

    
    return render_template("indexi.html", prediction_text = "The sentiment is {}".format(ans))

if __name__ == "__main__":
    flask_app.run(debug=True)

