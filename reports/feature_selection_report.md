# Relatório de Feature Selection

## Features Removidas por Multicolinearidade (|r| > 0.9)

- `has_review` ↔ `user_review_rate` (r=0.976)
- `price` ↔ `price_log` (r=1.000)
- `price` ↔ `user_avg_price` (r=0.982)
- `price` ↔ `product_avg_price` (r=0.997)
- `freight_value` ↔ `freight_value_log` (r=1.000)
- `freight_value` ↔ `user_avg_freight` (r=0.966)
- `price_log` ↔ `user_avg_price` (r=0.982)
- `price_log` ↔ `product_avg_price` (r=0.997)
- `freight_value_log` ↔ `user_avg_freight` (r=0.966)
- `days_since_reference` ↔ `user_recency_days` (r=-0.986)
- `user_avg_price` ↔ `product_avg_price` (r=0.979)
- `product_popularity` ↔ `product_unique_buyers` (r=1.000)

**Total removido:** 7 features

## Top 10 Mutual Information

| Feature | MI Score |
|---|---|
| review_score | 1.2054 |
| product_avg_review_score | 0.2928 |
| user_avg_freight | 0.0312 |
| user_avg_price | 0.0307 |
| product_avg_freight | 0.0215 |
| purchase_count | 0.0197 |
| user_total_purchases | 0.0170 |
| user_recency_days | 0.0135 |
| price_to_freight_ratio | 0.0130 |
| product_avg_price | 0.0129 |

## Top 10 Random Forest Importance

| Feature | Importance |
|---|---|
| review_score | 0.9707 |
| has_review | 0.0293 |
| user_avg_freight | 0.0000 |
| purchase_count | 0.0000 |
| category_target_enc | 0.0000 |
| days_since_reference | 0.0000 |
| user_recency_days | 0.0000 |
| price | 0.0000 |
| price_to_freight_ratio | 0.0000 |
| product_popularity | 0.0000 |

## Features Selecionadas (Total: 20)

- `price_log`
- `freight_value_log`
- `price_to_freight_ratio`
- `has_price_outlier`
- `days_since_reference`
- `is_weekend`
- `is_holiday_season`
- `category_target_enc`
- `category_frequency`
- `payment_type_credit_card`
- `payment_type_boleto`
- `payment_type_voucher`
- `payment_type_debit_card`
- `user_total_purchases`
- `user_avg_freight`
- `user_purchase_span_days`
- `user_recency_days`
- `product_popularity`
- `product_avg_review_score`
- `product_review_rate`
