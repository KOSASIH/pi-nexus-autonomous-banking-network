# course_management/__init__.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///course_management.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)

from .models import Course, Enrollment, User
from .schemas import CourseSchema, EnrollmentSchema, UserSchema
from .routes import course_routes, enrollment_routes, user_routes

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
