import time
import urllib.request
from summarizer import summarizer
from flask import Flask, render_template, redirect, jsonify, request
from flask_cors import CORS
import subprocess
import sys
import re
from wit import Wit


app = Flask(__name__)
CORS(app)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/parseHere/', methods = ['POST'])
def parse():
    text = request.form["pageData"]
    return parsify(text)


@app.route('/imageProcess/', methods = ['POST'])
def imgProc():
    url = request.form["pageData"]
    urllib.request.urlretrieve(url, "./temp")
    try:
        subprocess.call(["tesseract", "./temp", "out"])
        time.sleep(4)
    except:
        print("Give better file please")
    f = open('out.txt', 'r')
    return parsify(f.read())


def parsify(text):
    return summarizer(text) 




app.run(host='localhost', port=3000, debug=True)
