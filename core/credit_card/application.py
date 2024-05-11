import datetime
import random

class CreditCardApplication:
    def __init__(self, applicant_name, income, credit_score, employment_status, home_ownership):
        self.applicant_name = applicant_name
        self.income = income
        self.credit_score = credit_score
        self.employment_status = employment_status
        self.home_ownership = home_ownership
        self.application_date = datetime.datetime.now()
        self.application_number = random.randint(100000, 999999)

    def approve(self):
        if self.income > 50000 and self.credit_score > 700 and self.employment_status == "full-time" and self.home_ownership == "owned":
            return True
        else:
            return False

    def deny(self):
        return not self.approve()
