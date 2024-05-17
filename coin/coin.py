# coin.py
class DigitalAsset:
    def __init__(self, name: str, ticker: str, price: float):
        self._name = name
        self._ticker = ticker
        self._price = price

    @property
    def name(self) -> str:
        return self._name

    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        self._price = value
