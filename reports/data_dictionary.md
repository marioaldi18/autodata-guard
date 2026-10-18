# Data Dictionary

Generated from `src/autodataguard/data/schema.py`. Synthetic data only.

| Column | Type | Unit | Validation rules | Description |
|---|---|---|---|---|
| `order_id` | string | - | required; unique | Unique order identifier |
| `customer_id` | string | - | required | Customer identifier |
| `order_date` | date | - | required | Order date (ISO YYYY-MM-DD); must not be after the reference date |
| `product_category` | category | - | required; one of: Electronics, Clothing, Home, Beauty, Sports, Books | Product category |
| `quantity` | int | units | required; >= 1; <= 50 | Units ordered |
| `unit_price` | float | currency | required; >= 0.5; <= 5000 | Price per unit |
| `discount_pct` | int | percent | required; >= 0; <= 90 | Discount applied |
| `shipping_days` | int | days | optional; >= 0; <= 30 | Days from order to delivery |
| `payment_method` | category | - | required; one of: card, paypal, bank_transfer, cod | Payment method |
| `customer_age` | int | years | optional; >= 18; <= 100 | Customer age |
| `total_amount` | float | currency | required; >= 0 | Order total; must equal quantity * unit_price * (1 - discount_pct/100) within 0.011 |

Cross-column rule: `total_amount` must equal `quantity * unit_price * (1 - discount_pct/100)` within 0.011.
