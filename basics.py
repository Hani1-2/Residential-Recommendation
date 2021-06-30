# -*- coding: utf-8 -*-
"""
Created on Mon May  3 02:36:40 2021

@author: Hp
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
