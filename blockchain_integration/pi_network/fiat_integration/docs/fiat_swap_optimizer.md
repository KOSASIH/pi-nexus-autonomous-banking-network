Fiat Swap Optimizer
=====================

The Fiat Swap Optimizer module enables the optimization of fiat swaps.

**Methods**

* `optimize_fiat_swap(user_id, amount_pi, fiat_currency)`: Optimize a fiat swap.

**Usage**

```python
from fiat_swap_optimizer import FiatSwapOptimizer

fiat_swap_optimizer = FiatSwapOptimizer()
optimized_swap = fiat_swap_optimizer.optimize_fiat_swap("user123", 100, "USD")
