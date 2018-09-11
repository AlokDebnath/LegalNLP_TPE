import urllib.request
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
    urllib.request.urlretrieve(url, "./temp.jpg")
    subprocess.call(["gocr", "./temp.jpg", "-o", "file"])
    f = open('file', 'r')
    return parsify(f.read())


def parsify(text):
    return "hello world" + text




app.run(host='localhost', port=3000, debug=True)
