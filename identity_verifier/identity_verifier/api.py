from flask import Flask, request, jsonify
from identity_verifier.biometrics import BiometricVerifier

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    # Get the image from the request
    image = request.files['image'].read()

    # Verify the identity
    verifier = BiometricVerifier('facial_recognition')
    is_verified = verifier.verify(image)

    # Return the result
    return jsonify({'is_verified': is_verified})

@app.route('/biometrics', methods=['GET'])
def biometrics():
    # Return a list of available biometric verification methods
    return jsonify({'biometrics': ['facial_recognition']})

if __name__ == '__main__':
    app.run(debug=True)
