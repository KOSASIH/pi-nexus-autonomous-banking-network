from .bank import Bank

class Chase(Bank):
    def __init__(self, username, password):
        super().__init__(username, password)
        self.name = "Chase"
        self.url = "https://www.chase.com"
        self.login_url = "https://secure07c.chase.com/web/auth/login"
        self.accounts_url = "https://secure07c.chase.com/web/auth/dashboard#/accounts"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

    def login(self):
        # code for logging into Chase bank goes here

    def get_accounts(self):
        # code for getting account information from Chase bank goes here

    def get_transactions(self, account_id):
        # code for getting transaction history from Chase bank goes here
