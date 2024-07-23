import json
import os

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sidra_chain.db"
db = SQLAlchemy(app)


class SidraChainModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chain_id = db.Column(db.String(100), unique=True, nullable=False)
    chain_name = db.Column(db.String(100), nullable=False)


@app.route("/sidra_chain", methods=["GET"])
def get_sidra_chain():
    sidra_chain = SidraChainModel.query.all()
    return jsonify(
        [
            {"id": sc.id, "chain_id": sc.chain_id, "chain_name": sc.chain_name}
            for sc in sidra_chain
        ]
    )


@app.route("/sidra_chain", methods=["POST"])
def create_sidra_chain():
    data = request.get_json()
    sidra_chain = SidraChainModel(
        chain_id=data["chain_id"], chain_name=data["chain_name"]
    )
    db.session.add(sidra_chain)
    db.session.commit()
    return jsonify({"message": "Sidra Chain created successfully"})


if __name__ == "__main__":
    app.run(debug=True)
