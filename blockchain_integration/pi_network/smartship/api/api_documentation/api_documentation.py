import os
import json
from flask import Flask
from flask_swagger import Swagger

app = Flask(__name__)

swagger = Swagger(app, api_spec_url="/api/spec")

@app.route("/api/spec", methods=["GET"])
def api_spec():
    # Generate OpenAPI specification for API
    spec = {
        "openapi": "3.0.2",
        "info": {
            "title": "PI-Nexus Autonomous Banking Network API",
            "description": "API for PI-Nexus Autonomous Banking Network",
            "version": "1.0.0"
        },
        "paths": {
            "/api/login": {
                "post": {
                    "summary": "Login to API",
                    "requestBody": {
                        "description": "Login credentials",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "username": {"type": "string"},
                                        "password": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "access_token": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)

if __name__ == "__main__":
    app.run(debug=True)
