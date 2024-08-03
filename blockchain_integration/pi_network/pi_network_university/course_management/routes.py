# course_management/routes.py
from flask import Blueprint, request, jsonify
from . import app, db
from .models import Course, Enrollment, User
from .schemas import CourseSchema, EnrollmentSchema, UserSchema
from . import jwt

course_routes = Blueprint("course_routes", __name__)
enrollment_routes = Blueprint("enrollment_routes", __name__)
user_routes = Blueprint("user_routes", __name__)

@course_routes.route("/courses", methods=["GET"])
def get_courses():
    courses = Course.query.all()
    course_schema = CourseSchema(many=True)
    return jsonify(course_schema.dump(courses))

@course_routes.route("/courses", methods=["POST"])
@jwt_required
def create_course():
    data = request.get_json()
    course = Course(title=data["title"], description=data["description"])
    db.session.add(course)
    db.session.commit()
    return jsonify({"message": "Course created successfully"})

@enrollment_routes.route("/enrollments", methods=["GET"])
def get_enrollments():
    enrollments = Enrollment.query.all()
    enrollment_schema = EnrollmentSchema(many=True)
    return jsonify(enrollment_schema.dump(enrollments))

@enrollment_routes.route("/enrollments", methods=["POST"])
@jwt_required
def create_enrollment():
    data = request.get_json()
    enrollment = Enrollment(user_id=data["user_id"], course_id=data["course_id"])
    db.session.add(enrollment)
    db.session.commit()
    return jsonify({"message": "Enrollment created successfully"})

@user_routes.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()
    user_schema = UserSchema(many=True)
    return jsonify(user_schema.dump(users))

@user_routes.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    user = User(username=data["username"], email=data["email"], password=data["password"])
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created successfully"})

@user_routes.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"], password=data["password"]).first()
    if user:
        access_token = create_access_token(identity=user.username)
        return jsonify({"access_token": access_token})
    return jsonify({"message": "Invalid credentials"}), 401

app.register_blueprint(course_routes)
app.register_blueprint(enrollment_routes)
app.register_blueprint(user_routes)
