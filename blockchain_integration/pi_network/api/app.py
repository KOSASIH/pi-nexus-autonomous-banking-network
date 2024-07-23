from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from pi_network.core.config import Config
from pi_network.core.database import db
from pi_network.api.routes import api_blueprint

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

app.register_blueprint(api_blueprint)

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
