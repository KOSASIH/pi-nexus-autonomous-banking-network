from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'qlite:////tmp/test.db'
db = SQLAlchemy(app)

from routes import loan_processing, investment_management, risk_assessment

app.register_blueprint(loan_processing.bp)
app.register_blueprint(investment_management.bp)
app.register_blueprint(risk_assessment.bp)

if __name__ == '__main__':
    app.run(debug=True)
