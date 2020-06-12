import numpy as np
from flask import Flask, url_for, request, render_template, redirect, session #Flask class
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
import MySQLdb.cursors
import requests
import re
from datetime import datetime
import json
import pickle
import plotly

import plotly.graph_objects as go
import plotly.express as px

# import MySQLdb
# import mysql.connector
# import mysql.connector as mysql

# For Making parameters Configurable
with open('config.json', 'r') as c:
    params = json.load(c)["params"]
# local_server = True

#Initilize the flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'diabetes_dashboard'

# Intialize MySQL
mysql = MySQL(app)

def cal_mon_avg(month, year, formatteddate_list, glucolist):
    numofsam = 0
    Sum = 0
    for x in range(0, len(glucolist)):
        m = formatteddate_list[x][0] + formatteddate_list[x][1] + formatteddate_list[x][2]
        y = formatteddate_list[x][8] + formatteddate_list[x][9] + formatteddate_list[x][10] + formatteddate_list[x][11]
        if (m == month) and (y == year):
            Sum = glucolist[x] + Sum
            numofsam = numofsam + 1
    if numofsam > 0:
        Avg = (Sum / numofsam)
    else:
        Avg = 0
    return Avg

@app.route('/predict',methods=['GET','POST'])
def predict():
    #
    if 'loggedin' in session:
        # We need all the account info for the user so we can use it on the predict page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone() #Fetch the first row only
        print("Loggedin Block")
        #For Fasting Glucose
        if (request.method == 'POST'):
            print("If Block")
            fasting_gluco = float(request.form['fasting_glucose'])
            updated_features = [np.array([account['preg'], fasting_gluco, account['bmi'], account['age']])]  # Numpy Array of 4 Features
            print(updated_features)
            updated_prediction = model.predict(updated_features)
            updated_prediction_probability = np.amax(model.predict_proba(updated_features))
            updated_output = updated_prediction
            if updated_output == 1:
                return render_template('predict.html', prediction_text_1='According to K-Nearest Model, you are Diabetic with probability {}'.format(updated_prediction_probability), params=params, account=account)
            else:
                return render_template('predict.html', prediction_text_1='According to K-Nearest Model, you are Not Diabetic with probability {}'.format(updated_prediction_probability), params=params, account=account)
        #For Monthly Avgerged Glucose
        #Separate the Glucose Values of all Months
        else:
            print("Else Block")
            device_id = session['dev_id']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM glucose_value WHERE sensor = %s", (device_id,))
            tupleofdic = cursor.fetchall()
            listofdic = list(tupleofdic)
            glucolist = []
            datelist = []
            SNolist = []
            for dic in listofdic:
                glucose_ = dic['glucose']
                timestamp_ = dic['timestamp']
                id_ = dic['id']
                datelist.append(timestamp_)
                glucolist.append(glucose_)
                SNolist.append(id_)
            formatteddate_list = []
            for x in datelist:
                formatteddate_list.append(x.strftime("%b %d, %Y"))

            current_time = datetime.now()
            print(current_time)
            current_mon = current_time.strftime("%b")
            current_year = current_time.strftime("%Y")
            current_day = current_time.strftime("%d")
            print(current_mon)
            print(current_day)
            print(current_year)

            mon_year_text='For {} {} :'. format(current_mon,current_year)
            avg_glu_value = cal_mon_avg(current_mon, current_year, formatteddate_list, glucolist)
            gluco_text = 'Average Glucose Value = {} mg/dL'.format(avg_glu_value)
            updated_features = [
                np.array([account['preg'], avg_glu_value, account['bmi'], account['age']])]  # Numpy Array of 4 Features
            print(updated_features)
            updated_prediction = model.predict(updated_features)
            updated_prediction_probability = np.amax(model.predict_proba(updated_features))
            updated_output = updated_prediction
            #if updated_output == 1 and current_day == '30':
            if updated_output == 1:
                return render_template('predict.html',
                                       prediction_text_2='According to K-Nearest Model, you are Diabetic with probability {}'.format(
                                           updated_prediction_probability), params=params, account=account, current_mon =current_mon, current_year = current_year, avg_glu_value =avg_glu_value, mon_year_text =mon_year_text, gluco_text =gluco_text)
            # if updated_output == 0 and current_day == 30:
            if updated_output == 0:
                return render_template('predict.html',
                                       prediction_text_2='According to K-Nearest Model, you are Not Diabetic with probability {}'.format(
                                           updated_prediction_probability), params=params, account=account, current_mon =current_mon, current_year = current_year, avg_glu_value = avg_glu_value, mon_year_text =mon_year_text, gluco_text =gluco_text)
#
# Starting Page
@app.route('/', methods=['GET','POST'])
def newlogin():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE name = %s AND password = %s AND email = %s', (name, password, email))
        # Fetch one record and return result
        # List items are enclosed in square brackets [],
        # Tuple items in round brackets or parentheses (),
        # Dictionary items in curly brackets {}

        # account1 = cursor.fetchall() # Tuple of Dictionary
        # print(account1)
        # print(type(account1))
        account = cursor.fetchone() #Dictionary
        print(account)
        print(type(account))

        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['name'] = account['name']
            session['dev_id'] = account['dev_id']

            # Redirect to home page
            return redirect(url_for('index'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('newlogin.html', params=params, msg=msg)

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('name', None)
   session.pop('dev_id', None)
   # Redirect to login page
   return redirect(url_for('newlogin'))

@app.route('/newregister', methods=['GET', 'POST']) #Creating endpoints
def newregister():
    # Output message if something goes wrong...
    msg = ''
    # Check if "name", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'gender' in request.form and 'age' in request.form  and 'weight' in request.form and  'heightt' in request.form and 'dev_id' in request.form:
        # Create variables for easy access
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        gender = request.form['gender']
        age = request.form['age']
        weight = float(request.form['weight'])
        heightt = float(request.form['heightt'])
        bmi = weight / ((heightt) * (heightt))
        date = datetime.now()
        preg = request.form['preg']
        dev_id = request.form['dev_id']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', name):
            msg = 'name must contain only characters and numbers!'
        elif not name or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            insert_stmt = (
                "INSERT INTO accounts (name, email, password, gender, weight, heightt, age, bmi, preg, date, dev_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            data = (name, email, password, gender, weight, heightt, age, bmi, preg, date, dev_id)

            cursor.execute(insert_stmt, data)
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('newregister.html', msg=msg, params=params)
#
#3:08 AM 5/30/2020
# START POINT 6 from here from morning
#index.html in tutorial is autually newlogin.html whose route is /
# Open 4 Important webpages AND xammp
#
# http://localhost:5000/index - this will be the home page, only accessible for loggedin users
@app.route('/index') #Creating endpoints
def index():
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()  # Fetch the first row only

        #Grab sensor value from current user accounts!!!
        device_id = account['dev_id']

        # device_id = session['dev_id']


        #Tells the number of records already present in table for specific device id
        cursor.execute("SELECT COUNT(id) FROM glucose_value WHERE sensor = %s", (device_id,))
        old_rowcount = cursor.fetchone()["COUNT(id)"]
        print("Number of records present in glucose_value Table for", device_id, "=", old_rowcount )

        #Now specifing channels for each sensor alloted to user
        if device_id == "product1":
            thingspeak_cha = requests.get('https://api.thingspeak.com/channels/1077581/feeds.json')
        elif device_id == "product2":
            thingspeak_cha = requests.get('https://api.thingspeak.com/channels/1077581/feeds.json')
        elif device_id == "product3":
            thingspeak_cha = requests.get('https://api.thingspeak.com/channels/1077581/feeds.json')
        elif device_id == "product4":
            thingspeak_cha = requests.get('https://api.thingspeak.com/channels/1077581/feeds.json')
        elif device_id == "product5":
            thingspeak_cha = requests.get('https://api.thingspeak.com/channels/1077581/feeds.json')
        print(thingspeak_cha)

        dic_thingspeak = json.loads(thingspeak_cha.text)
        #Printing Dictionary
        print(dic_thingspeak)
        #Showing type as dictionary
        print(type(dic_thingspeak))
        # Actually it's a dict with two keys "channel" and "feeds".
        # The first one has another dict for value, and the second a list of dicts.

        list_of_dic_feeds = dic_thingspeak["feeds"]
        # Printing list of Dictionaries having channel data
        print(list_of_dic_feeds)
        # Showing Type as list
        print(type(list_of_dic_feeds))

        new_rowcountjson = len(list_of_dic_feeds)
        print("Number of records present in thingspeak channel =", new_rowcountjson)

        # Slicing list of data present in channel to get only the new records,
        # that are to be appended in MySQL glucose_value table
        new_newlist = list_of_dic_feeds[old_rowcount:]
        print("Number of new records present in thingspeak channel =",len(new_newlist))

        # Iterate over the list, for inserting new data of thingspeak channel into MySQL glucose_value Table:
        for entry in new_newlist:
            # Getting the value for the specific keys present in list of dic
            n_id = entry["entry_id"]
            n_glucose = entry["field1"]
            n_date = entry["created_at"]
            insert_stmt = (
                "INSERT INTO glucose_value (glucose, timestamp, sensor) "
                "VALUES (%s, %s, %s)"
            )
            data = (n_glucose, n_date, device_id)
            cursor.execute(insert_stmt, data)
            mysql.connection.commit()

        # Now Retrieving data from MYSQL glucose_value Table to plot graph and Table using Plotly library
        cursor.execute("SELECT * FROM glucose_value WHERE sensor = %s", (device_id,))
        tupleofdic = cursor.fetchall()
        listofdic = list(tupleofdic)
        glucolist = []
        datelist = []
        SNolist = []
        for dic in listofdic:
            glucose_ = dic['glucose']
            timestamp_ = dic['timestamp']
            id_ = dic['id']
            datelist.append(timestamp_)
            glucolist.append(glucose_)
            SNolist.append(id_)
        print(glucolist)
        print(datelist)
        print(SNolist)
        #For Starting S.No from 1 in the Table:
        print(len(SNolist))
        S_No_list = [*range(1, (len(SNolist) + 1), 1)]
        #
        formatteddate_list=[]
        for x in datelist:
            formatteddate_list.append(x.strftime("%b %d, %Y"))
        print(formatteddate_list)

        formattedtime_list =[]
        for x in datelist:
            formattedtime_list.append(x.strftime("%H:%M:%S"))
        print(formattedtime_list)
        #
        # For Finding id of last element present in glucose_value table of specific sensor,
        # for displaying the last updated date and glucose value
        cursor.execute('SELECT MAX(id) FROM glucose_value WHERE sensor = %s', (device_id,))
        max_id_dic = cursor.fetchone()
        max_id = max_id_dic["MAX(id)"]
        print(max_id)
        cursor.execute('SELECT * FROM glucose_value WHERE id = %s', (max_id,))
        glcodata = cursor.fetchone()  # Fetch the first row only
        print(glcodata)

        #Ploting graph using plotly format
        fig = go.Figure(data=go.Scatter(x=datelist, y=glucolist))
        fig.update_layout(
                          xaxis_title='Date',
                          yaxis_title='Glucose (mg/dL)')
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        # fig.show()
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        scatter = graphJSON

        #Ploting Table using plotly format
        headerColor = '#636EFA'
        rowEvenColor = 'rgb(179,205,227)'
        rowOddColor = 'white'
        fig1 = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>S.No</b>', '<b>Date</b>', '<b>Time</b>', '<b>Glucose (mg/dL)</b>'],
                line_color='darkslategray',
                fill_color=headerColor,
                align=['left', 'center'],
                font=dict(color='white', size=22)
            ),
            cells=dict(
                values=[
                    S_No_list,
                    formatteddate_list,
                    formattedtime_list,
                    glucolist,
                    ],
                line_color='darkslategray',
                # 2-D list of colors for alternating rows
                fill_color='rgb(203,213,232)',
                # fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor] * 5],
                align=['left', 'center'],
                font=dict(color='black', size=14),
                height=30
            ))
        ])

        # fig1.show()
        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        table = graphJSON
        #
        return render_template('index.html', account=account, params=params, plot=scatter, glcodata=glcodata, plot1=table)
    # User is not loggedin redirect to login page
    return redirect(url_for('newlogin'))

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone() #Fetch the first row only
        # Show the profile page with account info
        return render_template('profile.html', account=account, params=params)
    # User is not loggedin redirect to login page
    return redirect(url_for('newlogin'))

@app.route('/editprofile', methods=['GET', 'POST'])
def editprofile():
    namemsg = ''
    passwordmsg= ''
    agemsg= ''
    heighttmsg= ''
    weightmsg =''
    pregmsg = ''
    if 'loggedin' in session:
        if request.method == 'POST' and 'name' in request.form:
            name = request.form['name']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET name = %s WHERE id = %s"
            data = (name,session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            namemsg = 'You have successfully edited name!'
        elif request.method == 'POST' and 'password' in request.form:
            password = request.form['password']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET password = %s WHERE id = %s"
            data = (password, session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            passwordmsg = 'You have successfully edited password!'
        elif request.method == 'POST' and 'age' in request.form:
            age = request.form['age']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET age = %s WHERE id = %s"
            data = (age, session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            agemsg = 'You have successfully edited age!'
        elif request.method == 'POST' and 'heightt' in request.form:
            heightt = float(request.form['heightt'])
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET heightt = %s WHERE id = %s"
            data = (heightt, session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            heighttmsg = 'You have successfully edited height!'
        elif request.method == 'POST' and 'weight' in request.form:
            weight = float(request.form['weight'])
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET weight = %s WHERE id = %s"
            data = (weight, session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            weightmsg = 'You have successfully edited weight!'
        elif request.method == 'POST' and 'preg' in request.form:
            preg = request.form['preg']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            update_stmt = "UPDATE accounts SET preg = %s WHERE id = %s"
            data = (preg, session['id'])
            cursor.execute(update_stmt, data)
            mysql.connection.commit()
            pregmsg = 'You have successfully edited pregnancies number!'

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()  # Fetch the first row only

    if request.method == 'POST' and 'weight' in request.form or 'heightt' in request.form:

        UPweight = float(account['weight'])
        UPheightt = float(account['heightt'])
        bmi = UPweight / ((UPheightt) * (UPheightt))
        update_stmt = "UPDATE accounts SET bmi = %s WHERE id = %s"
        data = (bmi, session['id'])
        cursor.execute(update_stmt, data)
        mysql.connection.commit()
    return render_template('editprofile.html', account=account, name=session['name'], params=params, namemsg=namemsg, passwordmsg=passwordmsg, agemsg=agemsg, heighttmsg=heighttmsg, weightmsg=weightmsg, pregmsg=pregmsg)

app.run(debug=True) #To run on local host /Website running

