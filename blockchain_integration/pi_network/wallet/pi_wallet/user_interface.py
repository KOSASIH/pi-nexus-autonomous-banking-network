from flask import Flask, jsonify, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key_here"
Bootstrap(app)


class LoginForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=4, max=20)]
    )
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    submit = SubmitField("Login")


class TransactionForm(FlaskForm):
    amount = StringField("Amount", validators=[DataRequired()])
    currency = StringField("Currency", validators=[DataRequired()])
    recipient = StringField("Recipient", validators=[DataRequired()])
    submit = SubmitField("Send")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Login logic here
        return jsonify({"message": "Logged in successfully"})
    return render_template("login.html", form=form)


@app.route("/send", methods=["GET", "POST"])
def send_transaction():
    form = TransactionForm()
    if form.validate_on_submit():
        # Call multicurrency_support.py to process transaction
        from multicurrency_support import process_transaction

        result = process_transaction(
            form.amount.data, form.currency.data, form.recipient.data
        )
        return jsonify({"message": result})
    return render_template("send.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
