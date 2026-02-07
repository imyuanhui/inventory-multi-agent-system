
import json
from typing import List, Dict, Any

def parse_inventory_json(json_data: str) -> Dict[str, Any]:
    """
    Parses a JSON string containing product inventory data and extracts key insights.
    Focuses on identifying high-stock, low-stock, and potentially slow-moving items.

    Args:
        json_data: A JSON string with the inventory data in the specified format.

    Returns:
        A dictionary containing summarized inventory insights.
    """
    try:
        data = json.loads(json_data)
        products = data.get("products", [])

        total_products = len(products)
        total_variants = 0
        total_inventory_quantity = 0
        product_types: Dict[str, int] = {}
        vendor_counts: Dict[str, int] = {}

        high_stock_items: List[Dict[str, Any]] = []
        low_stock_items: List[Dict[str, Any]] = []

        # Define thresholds (can be made configurable later)
        HIGH_STOCK_THRESHOLD = 50
        LOW_STOCK_THRESHOLD = 10

        for product in products:
            product_type = product.get("product_type")
            if product_type:
                product_types[product_type] = product_types.get(product_type, 0) + 1

            vendor = product.get("vendor")
            if vendor:
                vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1

            variants = product.get("variants", [])
            total_variants += len(variants)

            for variant in variants:
                inventory_quantity = variant.get("inventory_quantity", 0)
                total_inventory_quantity += inventory_quantity

                item_summary = {
                    "product_title": product.get("title"),
                    "sku": variant.get("sku"),
                    "inventory_quantity": inventory_quantity,
                    "price": variant.get("price"),
                    "cost": variant.get("cost"),
                }

                if inventory_quantity >= HIGH_STOCK_THRESHOLD:
                    high_stock_items.append(item_summary)
                elif inventory_quantity <= LOW_STOCK_THRESHOLD:
                    low_stock_items.append(item_summary)

        insights = {
            "total_products": total_products,
            "total_variants": total_variants,
            "total_inventory_quantity": total_inventory_quantity,
            "product_type_distribution": product_types,
            "vendor_distribution": vendor_counts,
            "high_stock_items": high_stock_items,
            "low_stock_items": low_stock_items,
            "summary_statement": f"Analyzed {total_products} products with {total_variants} variants. Total inventory quantity: {total_inventory_quantity}. "
                               f"Found {len(high_stock_items)} high-stock items and {len(low_stock_items)} low-stock items."
        }

        return insights

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format provided."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def read_inventory_file(file_path: str) -> str:
    """
    Reads a JSON file from the given path and returns its content as a string.

    Args:
        file_path: The absolute or relative path to the JSON inventory file.

    Returns:
        A string containing the JSON content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an issue reading the file.
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"