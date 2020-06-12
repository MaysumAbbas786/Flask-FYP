18th April 2020
1. When Python is used as a server side scripting language for WEB DEVELOPMENT we have to use Djongo or Flask Framework
   In Flask Framework: We need to have create two folders:

   static -> Public Folder
   e.g https://www.codewithharry.com/static/  IT WILL WORK
 
   templates -> Private Folder (By Default)
   e.g https://www.codewithharry.com/templates/ IT WILL NOT WORK

2. Using Pycharm IDE for editing main.py
   We will run our app from main.py file using app.run (debug=TRUE)
 
   Using Sublime Text for editing Templates folder (.html file) which are my endpoints
   We will NOT run from Sublime Text nor for Live server of Visual Studio

3. For Sending any data from Python main.py to any .html end point use Templating
	Means in main.py:
			def function():
				python_variable= "XYZ"
				return render_template('index.html', html_variable=python_variable)
	Means in index.html:
			<p>Hi my name is {{html_variable}}</p>

4. Downloaded Bootstrap Template

5. Extracted Bootstrap Template and Jinja Templating
    We use a basic template and do changes on it like on admin dashboard
    How can you serve file from python progrm
    How can you use for loop

6. Starting to create a admin dashboard
    6.1 Fixing  url_for static ONLY change in href,script of index.html
        then everything is working on index.html endpoint
    6.2 ATM i am creating 4 endpoints index,login,register and password
    6.3 Now I have to set end points as currently they are not working
    6.4 Copy pasted all end points in my static folder and then
        search login in index.html and did the following change:
        FROM href="login.html" TO href="/login"
        Except href="login.html" Change rest
    6.5 But now same issue as no css, javascript etc were taken in login endpoint
    6.6 Create All 11 Endpoints in main.py aswell
    6.7 Now I will make layout.html for Template Inheritance

7. Template Inheritance In Jinja2
    7.1 jinja.pocoo.org | Footers, Headers, Javascript, css bohat se endpoints ki same
        hoti hain se we will automate them
        Focusing on 5 Endpoints
    7.2 For Changing Light side navigation to dark side navigation replace "sb-sidenav-light"
        of layout-sidenav-light.html with "sb-sidenav-dark"
    7.3 In charts.html, index.html, tables.html, layout-sidenav-light.html : <body>
        In layout-static.html :                                              <body class="sb-nav-fixed">
    7.4 For making sure every endpoint have same color of navigation I have executed 7.2
    7.5 For making sure every endpoint have same fixed navigation bar I have changed <body class="sb-nav-fixed">
        of layout-static.html with <body>
    7.6 IMPORTANT NOTES | CHANGES:
        layout-sidenav-light.html HAS DARK NAVIGATION BAR
        layout-static.html NAVIGATION BAR STICKS AT THE TOP WHEN SCROLLING
    7.7 From Line 1 to Line 97 of index.html I cut and pasted it into layout.html
    7.8 Then From Bottom of index.html till where the main body was finishing I cup and pasted into layout.html
    7.9 Then {% block body %} {% endblock %} in b/w of both pasted items in layout.html
    7.10 Then {% extends "layout.html" %}
              {% block body %}
                <main>
                </main>
                {% endblock %}
    7.11 This is how Template Inheritance is done so now do it for rest 4 endpoints aswell

19th April 2020
   Actual Bootstrap Front End Link:
       https://blackrockdigital.github.io/startbootstrap-sb-admin/dist/index.html
8. Connecting to Database
    8.1 Only error of favicon in concole, which we can solve later aswell. Which is icon present at top of chrome tab
    8.2 -Write xammp install in google and download from apachefriends
        -OPen Xampp Control Panel then Start Apache & MySQL
    8.3 -Write localhost/phpmyadmin/ in google
    8.4 -Edited layout.html a bit and then following is the target:
        ToDO:
        Fetch any random number from the database and show it on dashboard
    8.5 Now making a table in database:
        -Start xampp-control from Drive C
        -phpmyadmin is loaded and Table named Glucose_value is created
        -sno is made Primary key and Auto incremented as we will fetch using sno
        -phpmyadmin is basically a Ulility that helps us to manage our mysqldatabase
        It shows the Table and its content
        Helps us to change their content
   8.6 Now How to connect our Python app with database using flashsqlalchemy

9. Flask SQLAlchemy Tutorial (I am now shifting on my Diabetes Prediction App)

10. Making Parameters Configurable
    10.1 Make config.json
         I have written my URI etc
    10.2 Which Squlalchemy ki uri ko istimal karon ga



