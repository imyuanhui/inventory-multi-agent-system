# IRIS Multi-Agent Inventory Management System

## Production-Ready Implementation Guide with Google ADK & Gemini

**Philosophy**: Evals over vibes. Observability over print statements. Architecture over prompting.

**Reference**: [Production Agents Builder Day Guide V2](https://mirror-ladybug-8f7.notion.site/Production-Agents-Builder-Day-Guide-V2-2ff8ef702b5e80ecb4e2fc5ee26d6db6)

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Architecture Overview](#2-architecture-overview)
3. [Agent Implementations](#3-agent-implementations)
4. [Observability & Logging](#4-observability--logging)
5. [Root Orchestrator](#5-root-orchestrator)
6. [Evals](#6-evals)
7. [Production Checklist](#7-production-checklist)

---

## 1. Project Setup

### Create the project

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install google-adk posthog google-genai

# Scaffold the agent project
adk create iris_agents
```

### Directory Structure

ADK expects each agent module to have an `agent.py` with a `root_agent` variable.

```
iris_agents/
├── .env                          # API keys
├── agent.py                      # Root orchestrator (must export root_agent)
├── tools/
│   ├── inventory_tools.py        # Health classification tools
│   ├── similarity_tools.py       # Bundle recommendation tools
│   ├── pricing_tools.py          # Markdown pricing tools
│   └── supplier_tools.py         # Reorder alert tools
├── eval_sets/
│   ├── health_classifier.evalset.json
│   ├── bundle_recommender.evalset.json
│   ├── pricing_optimizer.evalset.json
│   └── reorder_alerter.evalset.json
├── test_config.json              # Eval thresholds
└── tests/
    └── test_agents.py            # Pytest eval integration
```

### Environment Variables

Create `iris_agents/.env`:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
POSTHOG_API_KEY=your_posthog_project_api_key
POSTHOG_HOST=https://eu.i.posthog.com
```

Get these (both free, no credit card):
- **Gemini API key**: https://aistudio.google.com → "Get API Key" → "Create API key"
- **PostHog**: https://posthog.com/signup (1M events/month free)

### Verify Setup

```bash
# Terminal chat
adk run iris_agents

# Visual Dev UI (recommended — gives you Trace tab + Eval tab)
adk web .
```

Open `http://localhost:8000`, select your agent, and chat. Use the **Trace** tab to inspect every tool call and model response.

---

## 2. Architecture Overview

### System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    IRIS Root Agent                            │
│  instruction: "Route inventory queries to the right agent"   │
│  sub_agents: [health, bundles, pricing, reorder]             │
└──────────────┬───────────────────────────────────────────────┘
               │  ADK handles routing via instruction + description
       ┌───────┼────────┬──────────────┬──────────────┐
       ▼       ▼        ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  Health    │ │  Bundle    │ │  Pricing   │ │  Reorder   │
│ Classifier │ │ Recommender│ │ Optimizer  │ │  Alerter   │
│            │ │            │ │            │ │            │
│ Tools:     │ │ Tools:     │ │ Tools:     │ │ Tools:     │
│ •calc_turn │ │ •find_sim  │ │ •get_price │ │ •check_lead│
│ •get_shelf │ │ •get_meta  │ │ •calc_elast│ │ •seasonal  │
│ •calc_vel  │ │ •bundle_sc │ │ •suggest_md│ │ •reorder_pt│
└────────────┘ └────────────┘ └────────────┘ └────────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                           │
                    PostHog auto-captures
                    all LLM calls + custom
                    tool events
```

### How Routing Works

ADK uses the root agent's `instruction` and each sub-agent's `description` to decide routing. No manual keyword matching needed. The LLM reads the descriptions and delegates.

| Sub-Agent | `description` (what ADK reads to route) |
|-----------|----------------------------------------|
| Health Classifier | "Classifies inventory SKUs as Healthy, At-Risk, or Critical" |
| Bundle Recommender | "Suggests complementary product pairings for slow-moving SKUs" |
| Pricing Optimizer | "Recommends markdown pricing strategies to clear inventory" |
| Reorder Alerter | "Generates preemptive reorder alerts based on stock levels" |

---

## 3. Agent Implementations

### Key ADK Rules

1. **Import from `google.adk.agents`** — not `google.adk`
2. **Tools are plain Python functions** — no decorator needed. ADK reads the **docstring** to decide which tool to call. Docstrings are critical.
3. **`instruction` is singular** — not `instructions`
4. **`description` is required** on sub-agents — ADK uses it for routing
5. **Keep instructions short and focused** — one job per agent, under 200 words
6. **Use `gemini-2.5-flash`** for routing and simple tasks. Only use Pro for complex reasoning.

---

### Agent 1: Inventory Health Classifier

**Purpose**: Assign health scores (Healthy / At-Risk / Critical) based on turnover ratios, shelf time, and velocity.

#### Tools

```python
# iris_agents/tools/inventory_tools.py

import time
import pandas as pd
from datetime import datetime

# In production, connect to database. For now, CSV.
INVENTORY_DF = pd.read_csv("data/sample_inventory.csv")

# PostHog client — initialized in agent.py, imported here
posthog_client = None

def set_posthog_client(client):
    global posthog_client
    posthog_client = client


def calculate_turnover(sku_id: str) -> dict:
    """Calculate the annual inventory turnover ratio for a given SKU.

    Use this tool to determine how quickly inventory sells relative
    to its average stock value. Higher ratios mean faster-moving stock.

    Args:
        sku_id: The SKU identifier (e.g., "SKU-FAST-001").

    Returns:
        dict with sku_id and turnover_ratio (annual).
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    turnover = round(sku_data["annual_cogs"] / sku_data["avg_inventory_value"], 2)
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "calculate_turnover",
            "agent_name": "health_classifier",
            "sku_id": sku_id,
            "latency_ms": latency,
            "success": True,
        })

    return {"sku_id": sku_id, "turnover_ratio": turnover}


def get_shelf_time(sku_id: str) -> dict:
    """Get the number of days a SKU has been sitting on the shelf since last restock.

    Use this tool to check how long inventory has been unsold.
    Longer shelf time indicates slower-moving or stagnant stock.

    Args:
        sku_id: The SKU identifier (e.g., "SKU-SLOW-002").

    Returns:
        dict with sku_id and days_on_shelf.
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    last_restock = datetime.fromisoformat(sku_data["last_restock_date"])
    days = (datetime.now() - last_restock).days
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "get_shelf_time",
            "agent_name": "health_classifier",
            "sku_id": sku_id,
            "latency_ms": latency,
            "success": True,
        })

    return {"sku_id": sku_id, "days_on_shelf": days}


def calculate_velocity(sku_id: str) -> dict:
    """Calculate average daily sales velocity for a SKU over the past 30 days.

    Use this tool to measure how fast a product sells per day.
    Higher velocity means the product is in active demand.

    Args:
        sku_id: The SKU identifier (e.g., "SKU-FAST-001").

    Returns:
        dict with sku_id and velocity_units_per_day.
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    velocity = round(sku_data["units_sold_30d"] / 30.0, 2)
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "calculate_velocity",
            "agent_name": "health_classifier",
            "sku_id": sku_id,
            "latency_ms": latency,
            "success": True,
        })

    return {"sku_id": sku_id, "velocity_units_per_day": velocity}
```

#### Agent Definition (in agent.py)

```python
from google.adk.agents import Agent
from tools.inventory_tools import calculate_turnover, get_shelf_time, calculate_velocity

health_classifier = Agent(
    model="gemini-2.5-flash",
    name="health_classifier",
    description="Classifies inventory SKUs as Healthy, At-Risk, or Critical based on turnover ratio, shelf time, and sales velocity.",
    instruction="""You are an inventory health analyst. Classify SKUs as Healthy, At-Risk, or Critical.

Decision criteria:
- Healthy: turnover > 6, shelf time < 60 days, velocity > 2 units/day
- At-Risk: turnover 3-6, shelf time 60-120 days, velocity 0.5-2 units/day
- Critical: turnover < 3, shelf time > 120 days, velocity < 0.5 units/day

For each SKU:
1. Call calculate_turnover to get the annual turnover ratio
2. Call get_shelf_time to get days on shelf
3. Call calculate_velocity to get daily sales velocity
4. Apply the criteria above and classify

Return a summary with counts per category (e.g., "3 Healthy, 1 At-Risk, 1 Critical").
Give 1-sentence reasoning per SKU. Do NOT suggest actions — only classify.""",
    tools=[calculate_turnover, get_shelf_time, calculate_velocity],
)
```

---

### Agent 2: Smart Product Bundle Recommender

**Purpose**: Identify slow-moving SKUs and suggest complementary product pairings using vector similarity.

#### Tools

```python
# iris_agents/tools/similarity_tools.py

import time
import numpy as np
import pandas as pd

EMBEDDINGS = np.load("data/product_embeddings.npy", allow_pickle=True).item()
PRODUCT_METADATA = pd.read_csv("data/sample_inventory.csv")

posthog_client = None

def set_posthog_client(client):
    global posthog_client
    posthog_client = client


def find_similar_products(sku_id: str, top_k: int = 5) -> list:
    """Find products most similar to the given SKU using vector similarity.

    Use this tool to discover complementary products that pair well
    with a slow-moving SKU for bundle recommendations.

    Args:
        sku_id: The target SKU to find similar products for.
        top_k: Number of similar products to return (default 5).

    Returns:
        List of dicts with sku_id and similarity_score for each match.
    """
    start = time.time()
    target = EMBEDDINGS[sku_id]
    results = []

    for candidate_sku, candidate_emb in EMBEDDINGS.items():
        if candidate_sku == sku_id:
            continue
        sim = float(np.dot(target, candidate_emb) / (
            np.linalg.norm(target) * np.linalg.norm(candidate_emb)
        ))
        if sim >= 0.7:
            results.append({"sku_id": candidate_sku, "similarity_score": round(sim, 3)})

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    results = results[:top_k]
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "find_similar_products",
            "agent_name": "bundle_recommender",
            "sku_id": sku_id,
            "matches_found": len(results),
            "latency_ms": latency,
            "success": True,
        })

    return results


def get_product_metadata(sku_id: str) -> dict:
    """Get product details including name, category, price, and brand.

    Use this tool to retrieve full product information for a SKU
    when building bundle recommendations.

    Args:
        sku_id: The SKU identifier.

    Returns:
        dict with name, category, price, and brand.
    """
    start = time.time()
    row = PRODUCT_METADATA[PRODUCT_METADATA["sku_id"] == sku_id].iloc[0]
    metadata = {
        "sku_id": sku_id,
        "name": row["product_name"],
        "category": row["category"],
        "price": float(row["price"]),
        "brand": row["brand"],
    }
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "get_product_metadata",
            "agent_name": "bundle_recommender",
            "sku_id": sku_id,
            "latency_ms": latency,
            "success": True,
        })

    return metadata


def calculate_bundle_score(primary_sku: str, complementary_sku: str) -> dict:
    """Calculate a composite bundle affinity score for two products.

    Combines vector similarity, price balance, and category affinity
    into a single score (0-100). Higher = better bundle candidate.

    Args:
        primary_sku: The slow-moving SKU to create a bundle for.
        complementary_sku: The candidate product to pair with.

    Returns:
        dict with primary_sku, complementary_sku, and bundle_score.
    """
    start = time.time()
    target_emb = EMBEDDINGS[primary_sku]
    comp_emb = EMBEDDINGS[complementary_sku]
    similarity = float(np.dot(target_emb, comp_emb) / (
        np.linalg.norm(target_emb) * np.linalg.norm(comp_emb)
    ))

    primary_price = PRODUCT_METADATA[PRODUCT_METADATA["sku_id"] == primary_sku].iloc[0]["price"]
    comp_price = PRODUCT_METADATA[PRODUCT_METADATA["sku_id"] == complementary_sku].iloc[0]["price"]
    price_ratio = max(primary_price, comp_price) / max(min(primary_price, comp_price), 0.01)
    price_factor = 1.0 if price_ratio < 2.0 else 0.7

    score = round((similarity * 0.6 + price_factor * 0.4) * 100, 1)
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "calculate_bundle_score",
            "agent_name": "bundle_recommender",
            "primary_sku": primary_sku,
            "complementary_sku": complementary_sku,
            "bundle_score": score,
            "latency_ms": latency,
            "success": True,
        })

    return {"primary_sku": primary_sku, "complementary_sku": complementary_sku, "bundle_score": score}
```

#### Agent Definition

```python
from tools.similarity_tools import find_similar_products, get_product_metadata, calculate_bundle_score

bundle_recommender = Agent(
    model="gemini-2.5-flash",
    name="bundle_recommender",
    description="Suggests complementary product pairings for slow-moving SKUs using vector similarity and bundle scoring.",
    instruction="""You are a product bundling specialist. Suggest complementary product pairings for slow-moving SKUs.

Workflow:
1. Call find_similar_products(sku_id) to get vector-similar candidates
2. For each candidate, call get_product_metadata to get name, category, price
3. Call calculate_bundle_score(primary_sku, candidate_sku) for a composite score
4. Rank by bundle_score descending and recommend the top 3-5 bundles

For each bundle, explain: "Customers who buy X often need Y because [reason]."
Include the similarity score and bundle score in your response.
Do NOT include the original slow-moving SKU as a recommendation.""",
    tools=[find_similar_products, get_product_metadata, calculate_bundle_score],
)
```

---

### Agent 3: Progressive Markdown (Pricing Optimizer)

**Purpose**: Recommend optimal markdown pricing based on shelf time, demand velocity, and seasonality.

#### Tools

```python
# iris_agents/tools/pricing_tools.py

import time
import pandas as pd
from datetime import datetime, timedelta

PRICE_HISTORY_DF = pd.read_csv("data/price_history.csv")
INVENTORY_DF = pd.read_csv("data/sample_inventory.csv")

posthog_client = None

def set_posthog_client(client):
    global posthog_client
    posthog_client = client


def get_price_history(sku_id: str, days_back: int = 90) -> list:
    """Retrieve historical pricing and sales data for a SKU.

    Shows how price changes affected demand over time.
    Use this to understand pricing patterns before recommending markdowns.

    Args:
        sku_id: The SKU identifier.
        days_back: Number of days of history to retrieve (default 90).

    Returns:
        List of dicts with date, price, and units_sold per day.
    """
    start = time.time()
    cutoff = datetime.now() - timedelta(days=days_back)
    history = PRICE_HISTORY_DF[
        (PRICE_HISTORY_DF["sku_id"] == sku_id) &
        (pd.to_datetime(PRICE_HISTORY_DF["date"]) >= cutoff)
    ]
    result = history[["date", "price", "units_sold"]].to_dict("records")
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "get_price_history",
            "agent_name": "pricing_optimizer",
            "sku_id": sku_id,
            "records_returned": len(result),
            "latency_ms": latency,
            "success": True,
        })

    return result


def calculate_elasticity(sku_id: str) -> dict:
    """Calculate the price elasticity of demand for a SKU.

    Elasticity > 1 means demand is sensitive to price changes (elastic).
    Elasticity < 1 means demand is relatively insensitive (inelastic).
    Use this to determine how aggressively to discount.

    Args:
        sku_id: The SKU identifier.

    Returns:
        dict with sku_id and elasticity value.
    """
    start = time.time()
    history = PRICE_HISTORY_DF[PRICE_HISTORY_DF["sku_id"] == sku_id].tail(30)

    if len(history) < 2:
        elasticity = 1.2  # Default when insufficient data
    else:
        price_changes = history["price"].pct_change().dropna()
        demand_changes = history["units_sold"].pct_change().dropna()
        valid = price_changes != 0
        if valid.any():
            elasticity = round(abs((demand_changes[valid] / price_changes[valid]).mean()), 2)
        else:
            elasticity = 1.2

    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "calculate_elasticity",
            "agent_name": "pricing_optimizer",
            "sku_id": sku_id,
            "elasticity": elasticity,
            "latency_ms": latency,
            "success": True,
        })

    return {"sku_id": sku_id, "elasticity": elasticity}


def suggest_markdown_price(sku_id: str, current_price: float, strategy: str, elasticity: float) -> dict:
    """Calculate an optimal markdown price based on strategy and price elasticity.

    Strategies: "aggressive" (30-50%), "moderate" (15-30%), "conservative" (5-15%).
    Higher elasticity means customers respond more to discounts.

    Args:
        sku_id: The SKU identifier.
        current_price: The current selling price.
        strategy: One of "aggressive", "moderate", or "conservative".
        elasticity: The price elasticity value from calculate_elasticity.

    Returns:
        dict with new_price, markdown_pct, and projected_velocity_increase.
    """
    start = time.time()
    ranges = {
        "aggressive": (0.30, 0.50),
        "moderate": (0.15, 0.30),
        "conservative": (0.05, 0.15),
    }
    lo, hi = ranges.get(strategy, (0.15, 0.30))

    if elasticity > 1.5:
        markdown_pct = hi
    elif elasticity > 1.0:
        markdown_pct = (lo + hi) / 2
    else:
        markdown_pct = lo

    new_price = round(current_price * (1 - markdown_pct), 2)
    velocity_increase = round(elasticity * markdown_pct * 100, 1)

    result = {
        "sku_id": sku_id,
        "new_price": new_price,
        "markdown_pct": round(markdown_pct * 100, 1),
        "projected_velocity_increase": velocity_increase,
    }
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "suggest_markdown_price",
            "agent_name": "pricing_optimizer",
            "sku_id": sku_id,
            "strategy": strategy,
            "markdown_pct": result["markdown_pct"],
            "latency_ms": latency,
            "success": True,
        })

    return result
```

#### Agent Definition

```python
from tools.pricing_tools import get_price_history, calculate_elasticity, suggest_markdown_price

pricing_optimizer = Agent(
    model="gemini-2.5-flash",
    name="pricing_optimizer",
    description="Recommends markdown pricing strategies to clear slow-moving inventory based on demand elasticity and shelf time.",
    instruction="""You are a pricing specialist. Recommend markdown strategies that maximize revenue while clearing slow inventory.

Workflow:
1. Call get_price_history(sku_id) to see historical pricing patterns
2. Call calculate_elasticity(sku_id) to measure price sensitivity
3. Call suggest_markdown_price(sku_id, current_price, strategy, elasticity) to get the optimized price

Strategy guidelines:
- Aggressive (30-50%): for critical inventory sitting >120 days
- Moderate (15-30%): for at-risk inventory sitting 60-120 days
- Conservative (5-15%): for seasonal items that will sell eventually

Always provide the recommended price, percentage markdown, and expected velocity increase.
If asked, provide 1-2 alternative strategies (more/less aggressive).""",
    tools=[get_price_history, calculate_elasticity, suggest_markdown_price],
)
```

---

### Agent 4: Pre-emptive Reorder Alerter

**Purpose**: Generate reorder alerts with supplier lead times and seasonal demand signals before stockouts.

#### Tools

```python
# iris_agents/tools/supplier_tools.py

import time
import json
import pandas as pd
from datetime import datetime

INVENTORY_DF = pd.read_csv("data/sample_inventory.csv")
with open("data/supplier_database.json") as f:
    SUPPLIER_DB = json.load(f)

posthog_client = None

def set_posthog_client(client):
    global posthog_client
    posthog_client = client


def calculate_reorder_threshold(sku_id: str) -> dict:
    """Calculate the minimum stock level at which a reorder should be placed.

    Uses daily sales velocity and supplier lead time to determine
    the reorder point with a 14-day safety buffer.

    Args:
        sku_id: The SKU identifier.

    Returns:
        dict with threshold, current_stock, daily_velocity, and needs_reorder flag.
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    current_stock = int(sku_data["current_stock_level"])
    daily_velocity = round(sku_data["units_sold_30d"] / 30.0, 2)

    supplier_id = sku_data["supplier_id"]
    lead_time = SUPPLIER_DB[supplier_id]["lead_time_days"]
    safety_buffer = 14

    threshold = int(daily_velocity * (lead_time + safety_buffer))

    result = {
        "sku_id": sku_id,
        "threshold": threshold,
        "current_stock": current_stock,
        "daily_velocity": daily_velocity,
        "needs_reorder": current_stock < threshold,
    }
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "calculate_reorder_threshold",
            "agent_name": "reorder_alerter",
            "sku_id": sku_id,
            "needs_reorder": result["needs_reorder"],
            "latency_ms": latency,
            "success": True,
        })

    return result


def check_lead_time(sku_id: str) -> dict:
    """Get the supplier delivery lead time for a SKU.

    Returns how many days it takes the supplier to deliver
    after an order is placed.

    Args:
        sku_id: The SKU identifier.

    Returns:
        dict with sku_id, supplier_id, and lead_time_days.
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    supplier_id = sku_data["supplier_id"]
    lead_time = SUPPLIER_DB[supplier_id]["lead_time_days"]

    result = {"sku_id": sku_id, "supplier_id": supplier_id, "lead_time_days": lead_time}
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "check_lead_time",
            "agent_name": "reorder_alerter",
            "sku_id": sku_id,
            "lead_time_days": lead_time,
            "latency_ms": latency,
            "success": True,
        })

    return result


def get_seasonal_demand_factor(sku_id: str) -> dict:
    """Get the seasonal demand multiplier for a SKU based on current date.

    Returns a factor: >1.0 means peak season (expect higher demand),
    <1.0 means off-season, 1.0 means normal. Use this to adjust
    reorder quantities for upcoming demand changes.

    Args:
        sku_id: The SKU identifier.

    Returns:
        dict with sku_id, category, month, and seasonal_factor.
    """
    start = time.time()
    sku_data = INVENTORY_DF[INVENTORY_DF["sku_id"] == sku_id].iloc[0]
    category = sku_data["category"]
    month = datetime.now().month

    seasonal_patterns = {
        "Apparel": {11: 1.8, 12: 2.2, 1: 0.7, 6: 0.9},
        "Electronics": {11: 2.0, 12: 2.5, 1: 0.6},
        "Home": {3: 1.3, 4: 1.4, 10: 1.2},
        "Sports": {5: 1.5, 6: 1.7, 7: 1.6},
    }
    factor = seasonal_patterns.get(category, {}).get(month, 1.0)

    result = {"sku_id": sku_id, "category": category, "month": month, "seasonal_factor": factor}
    latency = int((time.time() - start) * 1000)

    if posthog_client:
        posthog_client.capture("tool_call", {
            "tool_name": "get_seasonal_demand_factor",
            "agent_name": "reorder_alerter",
            "sku_id": sku_id,
            "seasonal_factor": factor,
            "latency_ms": latency,
            "success": True,
        })

    return result
```

#### Agent Definition

```python
from tools.supplier_tools import calculate_reorder_threshold, check_lead_time, get_seasonal_demand_factor

reorder_alerter = Agent(
    model="gemini-2.5-flash",
    name="reorder_alerter",
    description="Generates preemptive reorder alerts based on stock levels, supplier lead times, and seasonal demand.",
    instruction="""You are an inventory replenishment analyst. Identify SKUs that need reordering before they stock out.

Workflow:
1. Call calculate_reorder_threshold(sku_id) to get the minimum stock level
2. Call check_lead_time(sku_id) to get supplier delivery time
3. Call get_seasonal_demand_factor(sku_id) to check for demand spikes

Urgency levels:
- Immediate: current stock below threshold and stockout within lead time
- Within week: stockout expected within lead_time + 7 days
- Within month: stockout expected within lead_time + 30 days
- Not urgent: stock sufficient for 30+ days beyond lead time

Only report SKUs that need action. Include the projected stockout date,
recommended order quantity, and seasonal context in your response.""",
    tools=[calculate_reorder_threshold, check_lead_time, get_seasonal_demand_factor],
)
```

---

## 4. Observability & Logging

### PostHog Auto-Instrumentation

PostHog's Python SDK wraps your Gemini client and **auto-captures every LLM call** — inputs, outputs, tokens, cost, and latency. No manual logging needed for LLM calls.

```python
# At the top of iris_agents/agent.py

from posthog import Posthog
from posthog.ai.gemini import Client

# Initialize PostHog — auto-captures all LLM generations
posthog_client = Posthog(
    "<your_posthog_project_api_key>",
    host="https://eu.i.posthog.com"  # Use https://us.i.posthog.com for US
)

# Wrap the Gemini client — all LLM calls are now automatically logged
gemini_client = Client(
    api_key="your_gemini_api_key",
    posthog_client=posthog_client,
)

# Pass the PostHog client to tool modules for custom event logging
from tools.inventory_tools import set_posthog_client as set_inv_ph
from tools.similarity_tools import set_posthog_client as set_sim_ph
from tools.pricing_tools import set_posthog_client as set_price_ph
from tools.supplier_tools import set_posthog_client as set_supp_ph

set_inv_ph(posthog_client)
set_sim_ph(posthog_client)
set_price_ph(posthog_client)
set_supp_ph(posthog_client)
```

### What Gets Logged Automatically

PostHog's Gemini wrapper auto-captures:
- Every LLM generation (inputs, outputs, tokens, cost, latency)
- Model name and parameters
- Full trace chains across agent calls

### Custom Tool Events

Every tool function logs to PostHog via `posthog_client.capture("tool_call", {...})` with:

| Field | Description |
|-------|-------------|
| `tool_name` | Which tool was called |
| `agent_name` | Which agent owns this tool |
| `latency_ms` | Execution time in milliseconds |
| `success` | Whether the tool succeeded |
| `sku_id` | Domain-specific context |

### What PostHog Gives You

- **LLM Analytics dashboard** — every generation: inputs, outputs, tokens, cost, latency
- **Traces** — full chain of LLM calls and tool executions in a single agent run
- **Session replays** — watch real users interact with your agent
- **A/B test prompts** — feature flags to test prompt changes on a subset of users
- **Error tracking** — LLM errors auto-captured with alerts

**Docs:**
- AI Engineering: https://posthog.com/docs/ai-engineering
- LLM Analytics: https://posthog.com/docs/llm-analytics
- Gemini setup: https://posthog.com/docs/llm-analytics/installation/google

---

## 5. Root Orchestrator

### The Complete agent.py

```python
# iris_agents/agent.py

from google.adk.agents import Agent

# --- PostHog setup (see Section 4) ---
from posthog import Posthog
from posthog.ai.gemini import Client
import os

posthog_client = Posthog(
    os.environ.get("POSTHOG_API_KEY", ""),
    host=os.environ.get("POSTHOG_HOST", "https://eu.i.posthog.com"),
)

# Initialize PostHog on all tool modules
from tools.inventory_tools import set_posthog_client as set_inv_ph
from tools.similarity_tools import set_posthog_client as set_sim_ph
from tools.pricing_tools import set_posthog_client as set_price_ph
from tools.supplier_tools import set_posthog_client as set_supp_ph

set_inv_ph(posthog_client)
set_sim_ph(posthog_client)
set_price_ph(posthog_client)
set_supp_ph(posthog_client)

# --- Tool imports ---
from tools.inventory_tools import calculate_turnover, get_shelf_time, calculate_velocity
from tools.similarity_tools import find_similar_products, get_product_metadata, calculate_bundle_score
from tools.pricing_tools import get_price_history, calculate_elasticity, suggest_markdown_price
from tools.supplier_tools import calculate_reorder_threshold, check_lead_time, get_seasonal_demand_factor

# --- Sub-agents ---

health_classifier = Agent(
    model="gemini-2.5-flash",
    name="health_classifier",
    description="Classifies inventory SKUs as Healthy, At-Risk, or Critical based on turnover ratio, shelf time, and sales velocity.",
    instruction="""You are an inventory health analyst. Classify SKUs as Healthy, At-Risk, or Critical.

Decision criteria:
- Healthy: turnover > 6, shelf time < 60 days, velocity > 2 units/day
- At-Risk: turnover 3-6, shelf time 60-120 days, velocity 0.5-2 units/day
- Critical: turnover < 3, shelf time > 120 days, velocity < 0.5 units/day

For each SKU:
1. Call calculate_turnover to get the annual turnover ratio
2. Call get_shelf_time to get days on shelf
3. Call calculate_velocity to get daily sales velocity
4. Apply the criteria above and classify

Return a summary with counts per category. Give 1-sentence reasoning per SKU. Do NOT suggest actions — only classify.""",
    tools=[calculate_turnover, get_shelf_time, calculate_velocity],
)

bundle_recommender = Agent(
    model="gemini-2.5-flash",
    name="bundle_recommender",
    description="Suggests complementary product pairings for slow-moving SKUs using vector similarity and bundle scoring.",
    instruction="""You are a product bundling specialist. Suggest complementary product pairings for slow-moving SKUs.

Workflow:
1. Call find_similar_products(sku_id) to get vector-similar candidates
2. For each candidate, call get_product_metadata to get name, category, price
3. Call calculate_bundle_score(primary_sku, candidate_sku) for a composite score
4. Rank by bundle_score descending and recommend the top 3-5 bundles

For each bundle, explain why the pairing works. Include similarity and bundle scores.""",
    tools=[find_similar_products, get_product_metadata, calculate_bundle_score],
)

pricing_optimizer = Agent(
    model="gemini-2.5-flash",
    name="pricing_optimizer",
    description="Recommends markdown pricing strategies to clear slow-moving inventory based on demand elasticity and shelf time.",
    instruction="""You are a pricing specialist. Recommend markdown strategies that maximize revenue while clearing slow inventory.

Workflow:
1. Call get_price_history(sku_id) to see historical pricing patterns
2. Call calculate_elasticity(sku_id) to measure price sensitivity
3. Call suggest_markdown_price(sku_id, current_price, strategy, elasticity)

Strategies: aggressive (30-50%), moderate (15-30%), conservative (5-15%).
Always provide the recommended price, markdown percentage, and expected velocity increase.""",
    tools=[get_price_history, calculate_elasticity, suggest_markdown_price],
)

reorder_alerter = Agent(
    model="gemini-2.5-flash",
    name="reorder_alerter",
    description="Generates preemptive reorder alerts based on stock levels, supplier lead times, and seasonal demand.",
    instruction="""You are an inventory replenishment analyst. Identify SKUs that need reordering before they stock out.

Workflow:
1. Call calculate_reorder_threshold(sku_id) to get the minimum stock level
2. Call check_lead_time(sku_id) to get supplier delivery time
3. Call get_seasonal_demand_factor(sku_id) to check for demand spikes

Urgency levels: immediate, within_week, within_month, not_urgent.
Only report SKUs that need action. Include projected stockout date and order quantity.""",
    tools=[calculate_reorder_threshold, check_lead_time, get_seasonal_demand_factor],
)

# --- Root Orchestrator ---
# ADK uses the root agent's instruction + sub-agent descriptions to route automatically.
# No manual keyword matching needed.

root_agent = Agent(
    model="gemini-2.5-flash",
    name="iris_orchestrator",
    description="IRIS inventory management system that routes queries to specialist agents.",
    instruction="""You are the IRIS inventory management assistant. Route user queries to the right specialist agent.

Available specialists:
- health_classifier: for questions about inventory health, risk status, stock classification
- bundle_recommender: for product pairing, bundle suggestions, complementary items
- pricing_optimizer: for markdown pricing, discounts, clearance strategies
- reorder_alerter: for reorder timing, stockout prevention, supplier lead times

If a query spans multiple areas (e.g., "find at-risk items and suggest bundles"), delegate to each relevant agent in sequence.

You do NOT perform analysis yourself. Always delegate to the specialist agents.""",
    sub_agents=[health_classifier, bundle_recommender, pricing_optimizer, reorder_alerter],
)
```

That's it. One file. ADK handles the routing, session management, and agent delegation.

---

## 6. Evals

**If you skip evals, you don't have a production agent — you have a demo.**

### Step 1: Create Golden Datasets via Dev UI

The easiest way:

1. Run `adk web .`
2. Chat with your agent using test inputs
3. Go to the **Eval** tab → **"Add current session"**
4. Review and edit the expected trajectory and response

This creates `.evalset.json` files automatically.

### Step 2: Configure Eval Criteria

Create `iris_agents/test_config.json`:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": {
      "threshold": 0.8
    },
    "response_match_score": {
      "threshold": 0.5
    }
  }
}
```

- **`tool_trajectory_avg_score` at 0.8** — Agent must follow the expected tool sequence 80%+ of the time
- **`response_match_score` at 0.5** — Final response must share 50%+ word overlap (ROUGE-1) with golden answer

### Step 3: Run Evals

```bash
# Run evals for each agent's dataset
adk eval iris_agents iris_agents/eval_sets/health_classifier.evalset.json \
  --config_file_path=iris_agents/test_config.json \
  --print_detailed_results

adk eval iris_agents iris_agents/eval_sets/bundle_recommender.evalset.json \
  --config_file_path=iris_agents/test_config.json \
  --print_detailed_results

adk eval iris_agents iris_agents/eval_sets/pricing_optimizer.evalset.json \
  --config_file_path=iris_agents/test_config.json \
  --print_detailed_results

adk eval iris_agents iris_agents/eval_sets/reorder_alerter.evalset.json \
  --config_file_path=iris_agents/test_config.json \
  --print_detailed_results
```

### Test Cases to Create (Minimum 3 Per Agent)

**Health Classifier:**
1. Fast-moving healthy SKU → expect: `calculate_turnover` → `get_shelf_time` → `calculate_velocity` → classification "Healthy"
2. Slow-moving at-risk SKU → same tool sequence → classification "At-Risk"
3. Dead stock SKU → same tool sequence → classification "Critical"

**Bundle Recommender:**
1. Slow SKU with good matches → expect: `find_similar_products` → `get_product_metadata` (per match) → `calculate_bundle_score` (per match) → ranked recommendations
2. SKU with no similar products above threshold → `find_similar_products` returns empty → response says no bundles available
3. SKU with mixed-quality matches → tools called for each → only high-score bundles recommended

**Pricing Optimizer:**
1. Critical inventory, aggressive strategy → `get_price_history` → `calculate_elasticity` → `suggest_markdown_price` with "aggressive" → 30-50% markdown
2. At-risk inventory, moderate strategy → same tool sequence → 15-30% markdown
3. Seasonal item during holiday → same tools → conservative hold recommendation

**Reorder Alerter:**
1. Low stock SKU → `calculate_reorder_threshold` → `check_lead_time` → `get_seasonal_demand_factor` → "immediate" urgency alert
2. Well-stocked SKU → same tool sequence → no alert needed
3. Normal stock but peak season approaching → same tools → alert with seasonal adjustment

### Advanced Eval Criteria

Add these to `test_config.json` for deeper checks:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": {"threshold": 0.8},
    "response_match_score": {"threshold": 0.5},
    "final_response_match_v2": {"threshold": 0.7},
    "hallucinations_v1": {"threshold": 0.8}
  }
}
```

- **`final_response_match_v2`** — LLM judges semantic equivalence (not just word matching)
- **`hallucinations_v1`** — Checks if response is grounded in tool outputs. Essential.

### Pytest Integration (CI/CD)

```python
# iris_agents/tests/test_agents.py

from google.adk.evaluation.agent_evaluator import AgentEvaluator
import pytest

@pytest.mark.parametrize("eval_file", [
    "iris_agents/eval_sets/health_classifier.evalset.json",
    "iris_agents/eval_sets/bundle_recommender.evalset.json",
    "iris_agents/eval_sets/pricing_optimizer.evalset.json",
    "iris_agents/eval_sets/reorder_alerter.evalset.json",
])
def test_agent(eval_file):
    AgentEvaluator.evaluate(
        agent_module="iris_agents",
        eval_dataset_file_path=eval_file,
    )
```

Run with:

```bash
pytest iris_agents/tests/test_agents.py -v
```

### Debugging Failed Evals

1. **Use `--print_detailed_results`** to see exactly where the agent diverged
2. **Open `adk web .`** and check the **Trace** tab — inspect every tool call and model response
3. **If the answer is correct but worded differently**, switch to `final_response_match_v2` (LLM judge) instead of ROUGE
4. **If the agent makes wrong tool calls**, check your tool function docstrings — ADK uses these to decide which tool to call. Make them more specific.

---

## 7. Production Checklist

### Before Shipping

**Evals:**
- [ ] All 4 agents have ≥3 test cases each (created via Dev UI or manually)
- [ ] `adk eval` passes for all eval sets
- [ ] Trajectory score ≥0.8 for each agent
- [ ] Response score ≥0.5 for each agent
- [ ] Edge cases covered (empty results, unknown SKUs)

**Observability:**
- [ ] PostHog auto-capturing all LLM generations
- [ ] Every tool function logs `posthog.capture("tool_call", {...})`
- [ ] PostHog LLM Analytics dashboard shows traces
- [ ] Can trace a full request through root agent → sub-agent → tools

**Architecture:**
- [ ] Each agent has ONE specific job (no mega-prompts)
- [ ] Instructions are under 200 words per agent
- [ ] Tool docstrings clearly describe when to use each tool
- [ ] `description` set on every sub-agent for routing

**Safety:**
- [ ] API keys in `.env` / environment variables only (never hardcoded)
- [ ] `max_steps` set on agents to prevent infinite loops (ADK supports this)
- [ ] Input validation on tool functions (handle missing SKUs gracefully)

### Running in Production

```bash
# Dev UI for testing
adk web .

# Or wrap in a FastAPI endpoint
# (agent.py stays the same — import root_agent and call it)
```

### Cost Awareness

- Use `gemini-2.5-flash` for all routing and simple tasks (cheap, fast)
- Only use `gemini-2.5-pro` for agents that need complex reasoning
- Set `max_steps` on every agent to prevent runaway token usage
- Monitor token counts in PostHog LLM Analytics dashboard

---

## Quick Reference

| What | How |
|------|-----|
| Create project | `adk create iris_agents` |
| Run in terminal | `adk run iris_agents` |
| Visual Dev UI | `adk web .` |
| Run evals | `adk eval iris_agents <evalset.json> --config_file_path=test_config.json` |
| Import Agent | `from google.adk.agents import Agent` |
| Define tools | Plain Python functions with detailed docstrings |
| Agent routing | `sub_agents=` parameter + `description` on each sub-agent |
| PostHog LLM auto-capture | `from posthog.ai.gemini import Client` |
| PostHog custom events | `posthog.capture("tool_call", {tool_name, latency_ms, ...})` |
| Eval criteria config | `test_config.json` with `tool_trajectory_avg_score` + `response_match_score` |
| Pytest integration | `AgentEvaluator.evaluate(agent_module=..., eval_dataset_file_path=...)` |

---

## Resources

- **ADK Quickstart:** https://google.github.io/adk-docs/get-started/quickstart/
- **ADK Eval Docs:** https://google.github.io/adk-docs/evaluate/
- **ADK Eval Codelab:** https://codelabs.developers.google.com/adk-eval/instructions
- **ADK Crash Course:** https://codelabs.developers.google.com/onramp/instructions
- **ADK Sample Agents:** https://github.com/google/adk-samples
- **Gemini API Key (free):** https://aistudio.google.com
- **PostHog AI Engineering:** https://posthog.com/docs/ai-engineering
- **PostHog LLM Analytics:** https://posthog.com/docs/llm-analytics
- **PostHog Gemini Setup:** https://posthog.com/docs/llm-analytics/installation/google
