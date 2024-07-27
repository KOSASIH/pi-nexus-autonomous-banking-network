from typing import Dict

class SustainableSupplyChainContract:
    def __init__(self):
        self.suppliers: Dict[str, Dict[str, str]] = {}
        self.products: Dict[str, Dict[str, str]] = {}

    def register_supplier(self, supplier_id: str, name: str, location: str):
        # Register a new supplier
        self.suppliers[supplier_id] = {
            "name": name,
            "location": location,
            "products": []
        }

    def add_product(self, supplier_id: str, product_id: str, product_name: str):
        # Add a product to a supplier
        if supplier_id not in self.suppliers:
            raise ValueError("Supplier not found")
        self.suppliers[supplier_id]["products"].append(product_id)
        self.products[product_id] = {
            "supplier_id": supplier_id,
            "product_name": product_name,
            "sustainability_certifications": []
        }

    def add_sustainability_certification(self, product_id: str, certification: str):
        # Add a sustainability certification to a product
        if product_id not in self.products:
            raise ValueError("Product not found")
        self.products[product_id]["sustainability_certifications"].append(certification)

    def get_supplier_info(self, supplier_id: str):
        # Get the information of a supplier
        return self.suppliers.get(supplier_id, {})

    def get_product_info(self, product_id: str):
        # Get the information of a product
        return self.products.get(product_id, {})
