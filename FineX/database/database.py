from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

# Initialize database
db = SQLAlchemy()

# Initialize migrate
migrate = Migrate()


def init_app(app):
    """
    Initialize the database and migrate extensions for the given Flask app.
    """
    db.init_app(app)
    migrate.init_app(app, db)

    # Create database tables if not exists
    with app.app_context():
        db.create_all()


# Example model


class User(db.Model):
    """
    An example User model for the FineX project.
    """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"


# Example schema


class UserSchema(SQLAlchemyAutoSchema):
    """
    An example User schema for the FineX project.
    """

    class Meta:
        model = User
        load_instance = True
        include_relationships = True


user_schema = UserSchema()
users_schema = UserSchema(many=True)
