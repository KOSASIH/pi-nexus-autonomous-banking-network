from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/dbname'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)

from app import views, models
