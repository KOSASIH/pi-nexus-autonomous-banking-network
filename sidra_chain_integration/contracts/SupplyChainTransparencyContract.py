from typing import Dict, List

class SupplyChainTransparencyContract:
    def __init__(self):
        self.products: Dict[str, Dict[str, str]] = {}
        self.suppliers: Dict[str, Dict[str, List[str]]] = {}

    def register_product(self, product_id: str, product_name: str, supplier_id: str):
        # Register a new product
        self.products[product_id] = {
            "product_name": product_name,
            "supplier_id": supplier_id
        }
        self.suppliers[supplier_id] = self.suppliers.get(supplier_id, {})
        self.suppliers[supplier_id][product_id] = []

    def add_supplier(self, product_id: str, supplier_id: str, new_supplier_id: str):
        # Add a new supplier to a product
        if product_id not in self.products:
            raise ValueError("Product not found")
        if supplier_id not in self.suppliers:
            raise ValueError("Supplier not found")
        self.suppliers[supplier_id][product_id].append(new_supplier_id)

    def get_product_info(self, product_id: str):
        # Get the information of a product
        return self.products.get(product_id, {})

    def get_supplier_info(self, supplier_id: str):
        # Get the information of a supplier
        return self.suppliers.get(supplier_id, {})

    def get_product_suppliers(self, product_id: str):
        # Get the list of suppliers for a product
        return self.suppliers.get(self.products[product_id]["supplier_id"], {}).get(product_id, [])

    def get_supplier_products(self, supplier_id: str):
        # Get the list of products supplied by a supplier
        return list(self.suppliers.get(supplier_id, {}).keys())
