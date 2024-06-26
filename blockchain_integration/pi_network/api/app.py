from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from api.routes import auth_blueprint, wallet_blueprint

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pi_network.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

app.register_blueprint(auth_blueprint)
app.register_blueprint(wallet_blueprint)

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
