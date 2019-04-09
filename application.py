from flask import Flask, session,render_template,request,redirect,url_for
import os
import logging
#import ml libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
app = Flask(__name__)

#open the data that has been read from all the emails
with open('train.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)

vectorizer = CountVectorizer()
X_train_vector=vectorizer.fit_transform(X_train)
#train the model
mnb=MultinomialNB()
mnb.fit(X_train_vector,y_train)


email_list=[]

@app.route("/",methods=["POST","GET"])
def index():
    if request.method =="POST":
        email=request.form.get("email")
        if email=="":
            alert  = {"label":"warning","message":"Try Typing Something in the Email box !"}
            return render_template("index.html",email_list=email_list,alert=alert)
        em=email.split()
        result = mnb.predict(vectorizer.transform(em))
        pred=0
        for i in result:
            if i==0:
                pred+=1
        if pred/len(result)>0.5:
            final_result="spam"
        else:
            final_result="ham"
        if (final_result=="spam"):
            alert  = {"label":"danger","message":"This a spam"}
            return render_template("index.html",alert=alert,email_list=email_list)
        else:
            email_list.append(email[:100])
            alert  = {"label":"success","message":"Email Sent !"}
            return render_template("index.html",email_list=email_list,alert=alert)
    return render_template("index.html",email_list=email_list)
