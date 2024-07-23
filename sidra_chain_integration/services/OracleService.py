class OracleService:
    def __init__(self, oracle_model: OracleModel) -> None:
        self.oracle_model = oracle_model

    def update_price(self, asset: str, price: int) -> None:
        # Call the Oracle's updatePrice function using the service's model
        self.oracle_model.update_price(asset, price)

    def get_price(self, asset: str) -> int:
        # Query the Oracle's getPrice function using the service's model
        return self.oracle_model.get_price(asset)
