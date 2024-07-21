from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    credit_score = db.Column(db.Integer)
    income = db.Column(db.Integer)
    employment_history = db.Column(db.Integer)
    loan_amount = db.Column(db.Integer)

    def __init__(self, credit_score, income, employment_history, loan_amount):
        self.credit_score = credit_score
        self.income = income
        self.employment_history = employment_history
        self.loan_amount = loan_amount
