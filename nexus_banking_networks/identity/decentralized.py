import uport


class DecentralizedIdentity:

    def __init__(self, user_did):
        self.user_did = user_did

    def authenticate_user(self, user_credentials):
        # Authenticate user using decentralized identity management
        uport_client = uport.UportClient()
        authentication_response = uport_client.authenticate(
            self.user_did, user_credentials
        )
        return authentication_response
