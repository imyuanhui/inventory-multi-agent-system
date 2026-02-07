
from google.adk.agents.llm_agent import Agent
from google.adk.agents.workflow_agents import SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from typing import Dict, Any, List
import json

# Import the custom tool
from inventory_tools import parse_inventory_json


class InventoryAnalyst(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            name="InventoryAnalyst",
            description="Analyzes raw inventory JSON data to extract key insights like high-stock and low-stock items.",
            instruction="You are an expert inventory analyst. Your task is to analyze the provided inventory JSON data using the 'parse_inventory_json' tool. Identify high-stock and low-stock items based on the thresholds defined in the tool. Summarize the key findings and save them to the session state under the key 'inventory_insights'.",
            tools=[parse_inventory_json],
            model="gemini-2.0-flash", # Or your preferred Gemini model
            output_key="inventory_insights",
            **kwargs
        )

# Example of how the Orchestrator might use this agent (for context, not part of this agent's direct implementation)
# In a full multi-agent system, this would be part of the OrchestratorAgent
class StrategyOrchestrator(SequentialAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="StrategyOrchestrator",
            description="Orchestrates the inventory strategy generation process.",
            sub_agents=[
                InventoryAnalyst(),
                # MarketResearcher(), # Placeholder for future agents
                # StrategyArchitect(), # Placeholder for future agents
            ],
            **kwargs
        )

# This is the root agent that ADK will run. For a multi-agent system, this would typically be the orchestrator.
# For demonstrating the InventoryAnalyst, we can make it the root for now.
root_agent = InventoryAnalyst()

# Sample JSON data for testing (this would typically come from an external source or user input)
sample_inventory_json = '''
{
  "products": [
    {
      "id": 7001001001,
      "title": "Aran Knit Scarf - Forest Green",
      "vendor": "Aran Woollen Mills",
      "product_type": "Accessories",
      "created_at": "2024-06-15T09:00:00+00:00",
      "status": "active",
      "tags": "aran, knitwear, scarf, winter",
      "variants": [
        {
          "id": 40001001001,
          "sku": "ARAN-SCF-GRN",
          "price": "45.00",
          "cost": "18.00",
          "inventory_quantity": 28,
          "inventory_item_id": 50001001001
        }
      ]
    },
    {
      "id": 7001001002,
      "title": "Cashmere Sweater - Navy",
      "vendor": "Luxury Knitwear Co.",
      "product_type": "Apparel",
      "created_at": "2024-07-01T10:00:00+00:00",
      "status": "active",
      "tags": "cashmere, sweater, luxury",
      "variants": [
        {
          "id": 40001001002,
          "sku": "CASH-SWT-NVY",
          "price": "199.00",
          "cost": "80.00",
          "inventory_quantity": 5,
          "inventory_item_id": 50001001002
        },
        {
          "id": 40001001003,
          "sku": "CASH-SWT-NVY-L",
          "price": "199.00",
          "cost": "80.00",
          "inventory_quantity": 60,
          "inventory_item_id": 50001001003
        }
      ]
    },
    {
      "id": 7001001003,
      "title": "Wool Socks - Grey",
      "vendor": "Cozy Feet Inc.",
      "product_type": "Accessories",
      "created_at": "2024-06-20T11:00:00+00:00",
      "status": "active",
      "tags": "wool, socks, winter",
      "variants": [
        {
          "id": 40001001004,
          "sku": "WOOL-SCK-GRY",
          "price": "12.00",
          "cost": "4.00",
          "inventory_quantity": 120,
          "inventory_item_id": 50001001004
        }
      ]
    }
  ]
}
'''

# To run this agent, you would typically use `adk run` from the command line.
# The instruction given to the agent will guide it to use the `parse_inventory_json` tool.
# The output will be saved to the session state under 'inventory_insights'.

# For local testing without `adk run` (for demonstration purposes):
async def main():
    agent = InventoryAnalyst()
    # The agent's instruction will tell it to use the tool with the provided input.
    # In a real ADK run, the input would be passed to the agent via the ADK framework.
    # For this example, we simulate the input by directly calling the tool.
    insights = parse_inventory_json(sample_inventory_json)
    print("\n--- Inventory Insights ---")
    print(json.dumps(insights, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

