import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier1 = pickle.load(open('random.pkl','rb')) 
classifier2 = pickle.load(open('logistic.pkl','rb')) 
classifier3 = pickle.load(open('knear.pkl','rb')) 
classifier4 = pickle.load(open('gaussian.pkl','rb')) 
classifier5 = pickle.load(open('linear.pkl','rb')) 
classifier6 = pickle.load(open('decision.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------minors-------------------------------------------
@app.route('/minors')
def minors():
    return render_template('minors.html')


  
@app.route('/predict',methods=['GET'])
def predict():
    
    tweet = float(request.args.get('tweet'))
   
    
    
    Model = int(request.args.get('Model'))

    if Model=="0":
        prediction = classifier1.predict([[tweet]])

    elif Model=="1":
        prediction = classifier2.predict([[tweet]])
        
    elif Model=="2":
        prediction = classifier3.predict([[tweet]])

    elif Model=="3":
        prediction = classifier4.predict([[tweet]])
  
    elif Model=="4":
        prediction = classifier5.predict([[tweet]])

    else:
        prediction = classifier6.predict([[tweet]])
    
    import pandas as pd
    dataset= pd.read_csv('Data_tweet.csv')
        
    
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 25000)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, 3].values
        

    if prediction[0] == 1:
        
        return render_template('index.html', prediction_text='Predicted sentiment for given tweets is Positive ')
        
    else:
        return render_template('index.html', prediction_text='Predicted sentiment for given tweets is Negative ')


if __name__ == "__main__":
    app.run(debug=True)