import jwt
import requests


class OAuth:
    def __init__(self, client_id, client_secret, authorization_base_url, token_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_base_url = authorization_base_url
        self.token_url = token_url

    def get_authorization_url(self, redirect_uri, scope=None):
        payload = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
        }
        if scope:
            payload["scope"] = scope
        return self.authorization_base_url + "?" + requests.utils.urlencode(payload)

    def get_access_token(self, code, redirect_uri):
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(self.token_url, data=payload)
        return response.json()

    def refresh_access_token(self, refresh_token):
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(self.token_url, data=payload)
        return response.json()
