# Feature engineering and multicollinearity

## Feature engineering: 41 candidate predictors

Each row is one approved order. The prediction point is the order's approval, so every predictor must be known at approval time. Delivery dates, delivery durations and order status define the outcomes and are never predictors (`LEAKAGE_COLS` in `src/features.py`). The raw Olist tables (orders, order items, products, sellers, customers, payments, geolocation) are aggregated to order level by `build_feature_table`, and `add_extra_features` adds further approval-time features.

| Group | Built from | Candidate features (count) |
|---|---|---|
| Approval calendar | `order_approved_at` | day of week, day of month, month, ISO week of year (4) |
| Basket contents | order items + products | item count, distinct sellers, distinct product categories, summed product price, summed freight, summed revenue (price + freight), summed weight (7) |
| Seller location | sellers + geolocation | seller-to-customer great-circle distance (haversine, from median zip-prefix coordinates) as max, min, median and mean over the order's sellers; distinct seller zip prefixes, cities and states (7) |
| Payment | payments | total payment value; value and count for each payment type: boleto, credit card, debit card, voucher, not defined (11) |
| Promise and approval (extra) | orders | promised lead days (estimated delivery date shown at checkout − approval date); approval lag in hours (purchase → approval) (2) |
| Product catalogue (extra) | products | summed volume (length × height × width), mean number of photos (2) |
| Price structure (extra) | items, payments | freight-to-price ratio; maximum instalments chosen (2) |
| Seasonality and load (extra) | approval date, all orders | weekend flag, Black Friday period flag (20–30 Nov 2017), December flag; orders approved platform-wide in the 7 completed days before approval (4) |
| Categorical | customers, items + sellers | customer state; route type: all interstate / all same-state / mixed, from customer vs seller states (2) |

Total: 29 base numeric + 10 extra numeric + 2 categorical = **41 candidates**. One categorical, `payment_combination`, is derived during selection below to replace the per-method payment columns. Seller and product late-history features were explored in Phase 2 drafts and are not part of this set (see the Archive).

Many of these candidates describe the same quantity in different ways (revenue and its components; four summaries of the same distance; payment totals and their per-method breakdowns). That redundancy motivates the multicollinearity treatment below.

# Handling multicollinearity: shared feature selection

One frozen feature set is used by both tasks (late-delivery classification and days-from-promise regression), every cross-validation fold, every monthly evaluation and every model family. It reduces the **41 approval-time predictors** (39 numeric + 2 categorical) to **24 raw features** (21 numeric + 3 categorical), which reference coding expands to **53 encoded model columns**. Selection is done once, before modelling; it does not run inside model fitting.

All numbers below come from `results/feature_selection/` (written by `python -m src.feature_selection`, run from `phase2`). The same audit is run in section 4 of `01_data_features_multicollinearity.ipynb`, which writes these CSVs to `results/feature_selection/` and asserts that the result equals `SELECTED_NUM_COLS`.

## Why multicollinearity matters here

- **Linear and logistic models** (OLS and plain logistic regression): strongly correlated predictors give unstable coefficients (high variance, sign changes between folds) that cannot be read as the effect of one predictor with the others held fixed. An exact linear identity makes the design matrix singular, so coefficients are not unique.
- **Tree models** (Random Forest): prediction is largely unaffected, but split-based importance is shared arbitrarily between near-duplicates, so importance rankings understate each of them.

## Data used for the audit

- Only the Train rows of the June run: **45,972 classification** and **45,371 regression** orders, approved before 19 March 2018, with outcomes known when Validation begins. These are exactly the `train_df` rows of the two notebooks (checked: identical order IDs in both).
- Labels are used only to define these eligible populations. No Validation outcome, later-month outcome or model score is used to choose a predictor.
- Numeric VIF is computed on each task's median-imputed numeric matrix, with an intercept (correlation-matrix form). Constants and exact linear dependencies have infinite VIF.

**Exact-identity checks** (`exact_overlaps.csv`):

| Check | Classification rows | Regression rows | Share equal (tolerance 1e-8) |
|---|---:|---:|---|
| `order_revenue_sum` − (`order_pdt_price_sum` + `order_frieght_value_sum`) | 45,560 | 45,371 | 100% (max difference 4e-13) |
| `payment_value_sum` − sum of per-method payment values | 45,972 | 45,371 | 100% (max difference 2e-13) |
| `seller_dist_km_max` = `seller_dist_km_mean` | 45,308 | 45,121 | 98.96% (classification), 98.96% (regression) |
| `seller_dist_km_mean` = `seller_dist_km_min` | 45,308 | 45,121 | 98.96% (classification), 98.96% (regression) |

The 412 classification orders missing from the revenue check (45,972 − 45,560) have no item rows, so product price and revenue are missing; payment total is present for all of them.

## Funnel: 41 → 24 raw features

| Stage | Removed | Numeric remaining | Columns removed and rule |
|---|---:|---:|---|
| Start | – | 39 (+ 2 categorical = 41) | `NUM_COLS` (29) + `EXTRA_NUM_COLS` (10); categorical `customer_state`, `route_type` |
| 1. Constants | 2 | 37 | `payment_type_value_not_defined`, `payment_type_count_not_defined` (constant in the development-training rows; VIF infinite) |
| 2. Domain representative | 13 | 24 | `order_approved_week_of_year` (month and day of month represent timing; VIF 33,163 / 33,259); `order_revenue_sum` (exactly price + freight); `seller_dist_km_max`, `seller_dist_km_min`, `seller_dist_km_median` (`seller_dist_km_mean` represents distance; identical for ~99% of orders); 8 per-method payment columns `payment_type_value_{boleto, credit_card, debit_card, voucher}` and `payment_type_count_{boleto, credit_card, debit_card, voucher}` (represented by numeric `payment_value_sum` plus the new categorical `payment_combination`) |
| 3. Iterative joint VIF ≤ 5 | 3 | 21 | Repeatedly drop the column with the greatest VIF in either task (`payment_value_sum` exempt) until every VIF ≤ 5 in both: `order_pdt_price_sum` (VIF 65.93 classification / 23,818.87 regression), `seller_zip_code_prefix_count` (18.11 / 18.11), `seller_city_count` (5.91 / 5.91) |
| **Final** | **18** | **21** | + 3 categorical (`customer_state`, `route_type`, `payment_combination`) = **24 raw features** |

Before selection, several VIFs were infinite (payment columns and, in regression, price/freight/revenue) and the distance summaries exceeded 5 million. After selection the largest numeric VIF is **3.76 (classification) / 3.79 (regression)**, for `order_frieght_value_sum`. Full trace with the VIF of each column at its removal: `removal_trace.csv`.

`payment_combination` is built by `add_extra_features`: the sorted set of payment types used by an order, joined with `+` (e.g. `credit_card+voucher`; `no_payment` if none recorded). In the Train rows it has five levels: `credit_card`, `boleto`, `credit_card+voucher`, `voucher`, `debit_card`.

## Final features by meaning

Numeric VIF after selection, classification / regression (`numeric_vif_after.csv`).

**Timing (8 numeric)**

| Feature | VIF |
|---|---|
| `order_approved_day_of_week` | 2.38 / 2.38 |
| `order_approved_day_of_month` | 1.15 / 1.15 |
| `order_approved_month` | 1.81 / 1.82 |
| `is_weekend_approval` | 2.36 / 2.36 |
| `is_black_friday_period` | 2.33 / 2.33 |
| `is_december_peak` | 2.16 / 2.17 |
| `approval_lag_hours` | 1.07 / 1.07 |
| `orders_approved_prev_7d` | 1.98 / 1.98 |

**Basket (7 numeric)**

| Feature | VIF |
|---|---|
| `order_item_count` | 1.50 / 1.50 |
| `order_product_category_count` | 1.48 / 1.48 |
| `order_seller_count` | 2.29 / 2.29 |
| `seller_state_count` | 1.75 / 1.75 |
| `order_product_weight_g_sum` | 3.39 / 3.40 |
| `order_product_volume_cm3_sum` | 3.35 / 3.37 |
| `order_product_photos_mean` | 1.01 / 1.01 |

**Money (4 numeric + categorical `payment_combination`)**

| Feature | VIF |
|---|---|
| `payment_value_sum` | 1.64 / 1.66 |
| `order_frieght_value_sum` | 3.76 / 3.79 |
| `freight_to_price_ratio` | 1.30 / 1.31 |
| `payment_installments_max` | 1.19 / 1.20 |

**Logistics (2 numeric + categorical `customer_state`, `route_type`)**

| Feature | VIF |
|---|---|
| `seller_dist_km_mean` | 1.93 / 1.94 |
| `promised_lead_days` | 1.80 / 1.81 |

## Decisions

1. **Keep `payment_value_sum` rather than `order_pdt_price_sum`.** The two are near-duplicates (Spearman 0.99 in both tasks). Payment total is present for every order, including the 412 classification training orders with no item rows (price and revenue missing), so it needs no imputation. It is therefore exempt from the iterative VIF stage, which then removes product price.
2. **Keep both `seller_dist_km_mean` and `route_type`.** In the encoded design, distance has the largest VIF, **6.04 (classification) / 6.22 (regression)**, because it overlaps the route dummies: the median seller distance is about 100 km for all-same-state orders, 350 km for mixed routes and 663 km for all-interstate orders. They describe different things: distance measures how far the goods travel, route measures whether the shipment crosses a state border. This is the **single documented exception** to VIF ≤ 5 in the encoded design.
3. **Keep the nonlinear calendar flags.** `is_weekend_approval`, `is_black_friday_period` and `is_december_peak` are deterministic functions of day of week, month and day of month. Because the functions are nonlinear, VIF (a linear diagnostic) cannot detect the dependence; their VIFs are only 2.16–2.36. They are kept deliberately: they give the linear models step effects that tree models can find for themselves.

## Encoding: 24 raw features → 53 columns

Each row has exactly one level of each categorical, so a complete set of dummies for one categorical sums to the intercept column. With an intercept, full one-hot coding is therefore perfectly collinear (the **dummy-variable trap**) and coefficients are not unique. `make_preprocessor(..., reference_categories=True)` drops one reference level per categorical, so each dummy coefficient is a contrast with its reference.

| Block | Levels seen | Reference dropped | Encoded columns |
|---|---:|---|---:|
| Numeric (median-imputed) | – | – | 21 |
| `customer_state` | 27 | `SP` | 26 |
| `route_type` | 3 | `1. All interstate` | 2 |
| `payment_combination` | 5 | `credit_card` (single method) | 4 |
| **Total** | | | **53** |

Without reference coding the design would have 55 columns. With reference coding (`encoded_design_rank.csv`), the centred 53-column matrix has rank 53 in both tasks (full rank with an intercept). The largest encoded VIF is 6.04 (classification) / 6.22 (regression), for `seller_dist_km_mean` (decision 2). Next are `order_frieght_value_sum` (3.87 / 3.91), `order_product_weight_g_sum` (3.41 / 3.42), `order_product_volume_cm3_sum` (3.37 / 3.38) and `route_type_2. All same-state` (2.86 / 2.94). Every `customer_state` dummy is ≤ 1.74, and every `payment_combination` dummy is ≤ 1.79 (`reference_coded_vif.csv`).

## Transformations for linear models

Applied only to the OLS and logistic models (trees are invariant to monotone transforms; the Random Forest uses the 24 raw features). The transformations are pre-specified, not tested: log1p on the skewed inputs, sine/cosine for month and weekday, and the Mixed-route merge are always applied to every linear model, and there are no squared terms. Variable selection (CV-stepwise, BIC/AIC, Lasso) is the only outcome-driven step, and it runs on the transformed features. The skew screen is decided on Train in `01_data_features_multicollinearity.ipynb`, section 5; tables in `results/simple_to_complex/data/` (`01_skewness.csv`, `01_log1p_columns.csv`, `linear_design_vif.csv`).

- **log1p** when raw |skew| > 1 on Train, for the eight candidates listed in notebook 01. All eight qualify in both tasks (raw skew, classification): `payment_value_sum` 11.16, `order_frieght_value_sum` 8.99, `order_product_volume_cm3_sum` 8.22, `order_product_weight_g_sum` 6.70, `freight_to_price_ratio` 5.01, `approval_lag_hours` 4.61, `orders_approved_prev_7d` 1.73, `seller_dist_km_mean` 1.69. After log1p: 0.56, 1.19, -0.04, 0.38, 1.93, 0.97, 0.68, -1.10. Why: these money, size, time and load variables are non-negative with long right tails (raw skew 1.69 to 11.16). log1p compresses the tail so a few extreme orders do not dominate a linear fit, handles zeros, and gives per-doubling effects. The choice uses only the predictors' distribution on Train, never the outcome, so it adds no selection optimism; pre-specifying transformations is recommended by Harrell, *Regression Modeling Strategies*.
- **Cyclic calendar:** month and day of week become sine/cosine pairs; day of month stays numeric; the weekend, Black Friday and December flags are kept. Why: December is next to January and Sunday next to Monday, whereas a plain number imposes a false straight-line order and a jump at the wrap-around. Dummies would be exactly collinear with the weekend and December flags, and a sine/cosine pair captures a smooth cycle in two columns.
- **No squared terms.** Choosing polynomial terms by looking at the outcome would be another data-driven selection step, so selection stays the only one; each coefficient remains a single interpretable slope. The partial-residual plots in notebooks 02 and 03 are a diagnostic only: they show the pre-specified design leaves no strong curvature (small bends at the ends of the promised-lead-days range in classification), and any remaining nonlinearity is what the Random Forest stage tests.
- **Mixed route merged** into `1. All interstate` for all linear models: in Train, the 118 `3. Mixed` orders and the 227 orders with `seller_state_count >= 2` have zero late orders, so a plain logistic coefficient for them is not identified (quasi-separation).
- **VIF of the final linear design** (54 encoded columns, no aliased column): maximum 6.40 (classification) / 6.62 (regression), for `log_order_frieght_value_sum`; `log_payment_value_sum` 5.89 / 6.25 and `log_freight_to_price_ratio` 4.02 / 4.23. The logged money columns are nearly linearly related (log ratio is about log freight minus log price). The distance VIF falls to about 3 after logging and merging the route.

## Limitations

- **Single development sample.** Selection uses only the Train rows and is applied unchanged to every fold and month. Earlier historical CV folds therefore use a choice informed by later data; selection is not recomputed as of each fold date.
- **Linear diagnostic.** VIF detects linear dependence only (see decision 3). It does not measure nonlinear redundancy, which tree models can still exploit or split importance over.
- **Relative category effects.** Each dummy coefficient is relative to its reference level (`SP`, `1. All interstate`, single-method `credit_card`), not an absolute effect.
- **Fold-dependent width.** Encoder categories are learned within each training fold, so an earlier fold with fewer observed levels can have fewer than 53 columns. Reported importance names come from each fitted preprocessor.
- **Unseen categories.** A category not seen in training encodes as all zeros in its block, the same as the reference, so that model cannot distinguish it from the reference.

## Report-ready paragraph

*To limit multicollinearity, we selected one shared feature set before modelling, using only the Train orders of both tasks (45,972 classification and 45,371 regression orders) and no evaluation outcomes. From 41 approval-time predictors (39 numeric, 2 categorical), we removed 2 constant columns, then 13 redundant columns on domain grounds: an exact accounting identity (revenue = price + freight), seller-distance summaries identical to the mean for about 99% of orders, week of year, and eight per-method payment columns, which we replaced with total payment and a categorical payment-method combination. An iterative variance inflation factor (VIF) procedure across both tasks then removed product price, seller postcode count and seller city count. This left 21 numeric predictors with maximum VIF 3.79, plus customer state, route type and payment combination: 24 raw features in total. Dropping one reference level per categorical avoids the dummy-variable trap and gives a 53-column design that is full rank in both tasks. The only encoded VIF above 5 is seller distance (6.04 classification, 6.22 regression), which overlaps route type; we kept both because distance and crossing a state border are distinct logistics factors. Because VIF is linear, the deterministic calendar flags were kept deliberately as nonlinear effects. The selection is fixed from Train data and applied unchanged to all folds and months.*
