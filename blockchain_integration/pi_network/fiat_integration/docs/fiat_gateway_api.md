Fiat Gateway API
================

The Fiat Gateway API client enables the interaction with various fiat gateways for swapping Pi coins to fiat money.

**Methods**

* `get_fiat_exchange_rate(fiat_currency)`: Get the current fiat exchange rate for the specified fiat currency.
* `execute_swap(amount_fiat, fiat_currency)`: Execute a fiat swap for the specified amount and fiat currency.

**Usage**

```python
from fiat_gateway_api import FiatGatewayAPI

fiat_gateway_api = FiatGatewayAPI()
exchange_rate = fiat_gateway_api.get_fiat_exchange_rate("USD")
swap_response = fiat_gateway_api.execute_swap(100, "USD")
