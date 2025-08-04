from flask import Flask, render_template
from flask_security import (
    RoleMixin,
    Security,
    SQLAlchemyUserDatastore,
    UserMixin,
    login_required,
)

from config import Config
from models import Role, User, db

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)


@app.route("/")
@login_required
def index():
    return render_template("index.html")


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
