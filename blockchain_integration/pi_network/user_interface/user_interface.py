# user_interface.py

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key_here"
Bootstrap(app)

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=50)])
    submit = SubmitField("Login")

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=50)])
    submit = SubmitField("Register")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Login logic here
        return jsonify({"status": "success"})
    return render_template("login.html", form=form)

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Register logic here
        return jsonify({"status": "success"})
    return render_template("register.html", form=form)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    # Dashboard logic here
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)

# templates/index.html

<!DOCTYPE html>
<html>
  <head>
    <title>Home</title>
  </head>
  <body>
    <h1>Welcome to Pi-Nexus!</h1>
    <p>Log in or register to access your account.</p>
    <a href="{{ url_for('login') }}">Login</a>
    <a href="{{ url_for('register') }}">Register</a>
  </body>
</html>

# templates/login.html

<!DOCTYPE html>
<html>
  <head>
    <title>Login</title>
  </head>
  <body>
    <h1>Login</h1>
    <form action="" method="post">
      {{ form.hidden_tag() }}
      <label for="username">Username:</label>
      {{ form.username(size=20) }}
      <br>
      <label for="password">Password:</label>
      {{ form.password(size=20) }}
      <br>
      {{ form.submit() }}
    </form>
  </body>
</html>

# templates/register.html

<!DOCTYPE html>
<html>
  <head>
    <title>Register</title>
  </head>
  <body>
    <h1>Register</h1>
    <form action="" method="post">
      {{ form.hidden_tag() }}
      <label for="username">Username:</label>
      {{ form.username(size=20) }}
      <br>
      <label for="email">Email:</label>
      {{ form.email(size=20) }}
      <br>
      <label for="password">Password:</label>
      {{ form.password(size=20) }}
      <br>
      {{ form.submit() }}
    </form>
  </body>
</html>

# templates/dashboard.html

<!DOCTYPE html>
<html>
  <head>
    <title>Dashboard</title>
  </head>
  <body>
    <h1>Dashboard</h1>
    <p>Welcome, {{ current_user.username }}!</p>
    <ul>
      <li><a href="#">Account Settings</a></li>
      <li><a href="#">Transaction History</a></li>
      <li><a href="#">Transfer Funds</a></li>
    </ul>
  </body>
</html>
