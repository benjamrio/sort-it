import cv2
from wtforms import Form, StringField, TextAreaField, DecimalField, IntegerField, FloatField, PasswordField, validators
from flask import Flask, render_template, redirect, url_for, session, request, flash
from detection.live_haarcascade_detection import *
from detection.haarcascade_detection import *
from convolutional_neural_network.prediction import *


app = Flask(__name__)

# Index
@ app.route('/')
def index():
    return render_template('home.html')


# About
@ app.route('/about')
def about():
    return render_template('about.html')



# Project
@ app.route('/project', methods=['GET', 'POST'])
def project():
    form = ParamForm(request.form)

    if request.method == "POST":
        #recupération des donées du form
        scaleFactor = form.scaleFactor.data
        minNeighbours = form.minNeighbours.data
        pathToImage= form.pathToImage.data

        if request.form['submit_button']=="Live Capture":
            return(startCapture(scaleFactor,minNeighbours))

        elif request.form['submit_button']=="Upload Picture":
            return(uploadPicture(pathToImage,scaleFactor,minNeighbours))
        
    return render_template("project.html",form=form)


#Parameters Form Class
class ParamForm(Form):
    scaleFactor = FloatField('Scale Factor', [validators.InputRequired(),validators.Length(min=1)])
    minNeighbours= IntegerField('Minimum Neighbours',[validators.InputRequired(), validators.Length(min=1)])
    pathToImage= StringField("Path to Image",[validators.Length(min=2)]) #only used for "upload picture"




# Start Capture
def startCapture(scaleFactor,minNeighbours):
    live_detection(scaleFactor,minNeighbours)
    (materiel,poubelle) =predict_path('Data/newPicture.jpg')
    print(materiel,poubelle)                             #On affichera également les résultats des prédictions dans la console
   
    flash('The picture is predicted to be ' + materiel +", therefore it goes in " + poubelle + " trash.", 'message')
    return(redirect(url_for('project')))



#Upload Picture
def uploadPicture(path,scaleFactor,minNeighbours):
    haarcascade_detection(path,scaleFactor,minNeighbours)
    (materiel, poubelle)=predict_path('Data/newPicture.jpg')
    print(materiel,poubelle)                              #On affichera également les résultats des prédictions dans la console
    
    flash('The picture is predicted to be ' + materiel +", therefore it goes in " + poubelle + " trash.", 'message')
    return(redirect(url_for('project')))





if __name__ == "__main__":
    app.secret_key = 'johncagnol'
    app.run()
