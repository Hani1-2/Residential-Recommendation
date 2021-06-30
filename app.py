# -*- coding: utf-8 -*-
"""
Created on Mon May  3 02:37:19 2021

@author: Hp
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('iri.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    arr = []
    if request.method == 'POST':
        data1 = request.form['a']
        if data1 == "Parks,Restaraunts,Grocery Stores":
            data1 = 14
        if data1 == "Grocery Stores,Mosques,Recreational Clubs":
            data1 = 2
        if data1 == "Parks,Restaraunts,Grocery Stores":
            data1 = 12
        if data1 == "Parks,Grocery Stores,Recreational Clubs":
            data1 = 6
        if data1 == "Parks,Mosques,Recreational Clubs":
            data1 = 9
        if data1 == "Restaraunts,Grocery Stores,Recreational Clubs":
            data1 = 16
        if data1 == "Parks,Grocery Stores,Mosques":
            data1 = 5
        if data1 == "Restaraunts,Grocery Stores,Mosques":
            data1 = 17
        if data1 == "Parks,Restaraunts,Recreational Clubs":
            data1 = 15
        else:
            data1 = 13
        arr.append(data1)
        data2 = request.form['b']
        if (data2 == "Social, Noisy, Public"):
            data2 = 1
        elif (data2 == "Social, Quiet, Private"):
            data2 = 2
        else:
            data2 = 0
        arr.append(data2)
        data3 = request.form['c']
        arr.append(data3)
        # data3 means Nature
        data4 = request.form['d']
        if (data4 == "Extrovert"):
            data4 = 1
        elif (data4 == "Introvert"):
            data4 = 2
        else:
            data4 = 0
        arr.append(data4)
    # data5 means Mental Peace
        data5 = request.form['e']
        if data5 == "Yes":
            data5 = 1
        else:
            data5 = 0
        arr.append(data5)
    # data6 means Reaction on lack  of somthing
        data6 = request.form['f']
        if data6 == "Pursue something else":
            data6 = 1
        else:
            data6 = 0
        arr.append(data6)
        data7 = request.form['g']
        if data7 == "Socialize":
            data7 = 2
        elif data7 == "Kill time":
            data7 = 0
        elif data7 == "Pursue hobbies (Sports, Reading, Painting etc)":
            data7 = 1
        else:
            data7 = 3
        arr.append(data7)
        data8 = request.form['h']
        if data8 == "Once or twice a week":
            data8 = 2
        elif data8 == "Once or twice a month":
            data8 = 1
        elif data8 == "Rarely":
            data8 = 3
        else:
            data8 = 0
        arr.append(data8)
        data9 = request.form['i']
        if data9 == "Never talk about that topic":
            data9 = 1
        elif data9 == "Try to change the other person's view":
            data9 = 3
        elif data9 == "Talk about the topic respecting eachother's views":
            data9 = 2
        else:
            data9 == 0
        arr.append(data9)
        data10 = request.form['j']
        if data10 == "Desi":
            data10 = 1
        elif data10 == "Ginsoy":
            data10 = 3
        elif data10 == "KFC":
            data10 = 4
        elif data10 == "Dominos":
            data10 = 2
        else:
            data10 = 0
        arr.append(data10)
        arr1 = np.array([arr])
        pred = model.predict(arr1)
        return render_template("house.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)
