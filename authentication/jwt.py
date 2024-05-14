import jwt
import datetime

class JWT:
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def encode(self, payload):
        payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode(self, token):
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
