from app import db

class Currency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    exchange_rate = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<Currency {self.name}>'
