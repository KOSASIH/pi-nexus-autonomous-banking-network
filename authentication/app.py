# app.py
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
jwt = JWTManager(app)


@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    # Authenticate user
    if authenticate_user(username, password):
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token})


@app.route("/protected", methods=["GET"])
@jwt_required
def protected():
    return jsonify({"message": "Hello, authenticated user!"})


if __name__ == "__main__":
    app.run(debug=True)
