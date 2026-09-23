-- ===========================================================================
-- 03_business_queries.sql
-- The BI layer: every churn-by-segment number in the notebooks and the
-- dashboard comes from one of these named queries, so the SQL a reviewer
-- reads is exactly the SQL that produced the chart.
--
-- Format: each query starts with a "-- name: <query_name>" line and runs to
-- the next one. src/data_preprocessing.py::run_named_query() loads them.
-- ===========================================================================

-- name: kpi_overview
SELECT
    COUNT(*)                                            AS customers,
    SUM(churn)                                          AS churned,
    ROUND(AVG(churn) * 100, 2)                          AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2)                      AS monthly_revenue,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2)
                                                        AS monthly_revenue_lost,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END)
          / SUM(monthly_charges) * 100, 2)              AS revenue_churn_pct,
    ROUND(AVG(tenure), 1)                               AS avg_tenure_months
FROM customers;

-- name: churn_by_contract
SELECT
    contract                                            AS segment,
    COUNT(*)                                            AS customers,
    SUM(churn)                                          AS churned,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) * 12, 0)
                                                        AS annual_revenue_lost
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_bucket
SELECT
    tenure_bucket                                       AS segment,
    COUNT(*)                                            AS customers,
    SUM(churn)                                          AS churned,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charges
FROM customer_features
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

-- name: churn_by_payment_method
SELECT
    payment_method                                      AS segment,
    COUNT(*)                                            AS customers,
    SUM(churn)                                          AS churned,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- name: churn_by_internet_service
SELECT
    internet_service                                    AS segment,
    COUNT(*)                                            AS customers,
    SUM(churn)                                          AS churned,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charges
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- name: churn_by_protection
-- Security and tech support are the add-ons that most change behaviour.
SELECT
    CASE WHEN internet_service = 'No' THEN 'No internet'
         WHEN online_security = 'Yes' AND tech_support = 'Yes' THEN 'Security + Support'
         WHEN online_security = 'Yes' OR  tech_support = 'Yes' THEN 'One of the two'
         ELSE 'Neither'
    END                                                 AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct
FROM customers
GROUP BY segment
ORDER BY churn_rate_pct DESC;

-- name: churn_by_senior
SELECT
    CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-senior' END AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct
FROM customers
GROUP BY senior_citizen
ORDER BY churn_rate_pct DESC;

-- name: churn_by_service_count
SELECT
    service_count                                       AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct
FROM customer_features
GROUP BY service_count
ORDER BY service_count;

-- name: churn_contract_x_internet
-- The two strongest categorical drivers, crossed. Month-to-month fiber
-- customers are the single riskiest cell in the book.
SELECT
    contract,
    internet_service,
    COUNT(*)                                            AS customers,
    ROUND(AVG(churn) * 100, 1)                          AS churn_rate_pct
FROM customers
GROUP BY contract, internet_service
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_month
-- Monthly churn curve plus a running share of all churners, using window
-- functions: how front-loaded is churn?
WITH by_month AS (
    SELECT tenure,
           COUNT(*)   AS customers,
           SUM(churn) AS churned
    FROM customers
    GROUP BY tenure
)
SELECT
    tenure,
    customers,
    churned,
    ROUND(churned * 100.0 / customers, 1)               AS churn_rate_pct,
    ROUND(SUM(churned) OVER (ORDER BY tenure) * 100.0
          / SUM(churned) OVER (), 1)                    AS cumulative_share_of_churners_pct
FROM by_month
ORDER BY tenure;

-- name: top_revenue_churners_by_contract
-- The three most valuable customers lost in each contract type.
SELECT contract, customer_id, tenure, monthly_charges, rank_in_contract
FROM (
    SELECT contract, customer_id, tenure, monthly_charges,
           ROW_NUMBER() OVER (PARTITION BY contract
                              ORDER BY monthly_charges DESC) AS rank_in_contract
    FROM customers
    WHERE churn = 1
)
WHERE rank_in_contract <= 3
ORDER BY contract, rank_in_contract;

-- name: retention_targets
-- Needs the customer_scores table written by the scoring step. Active
-- customers only (churned ones are already gone), ranked by expected loss.
SELECT
    s.customer_id,
    s.churn_probability,
    s.risk_level,
    s.expected_loss,
    s.expected_net_gain,
    s.top_reasons,
    c.contract,
    c.tenure,
    c.monthly_charges
FROM customer_scores s
JOIN customers c USING (customer_id)
WHERE c.churn = 0
ORDER BY s.expected_loss DESC;

-- name: risk_summary
SELECT
    s.risk_level,
    COUNT(*)                                            AS customers,
    ROUND(AVG(s.churn_probability) * 100, 1)            AS avg_churn_probability_pct,
    ROUND(SUM(s.expected_loss), 0)                      AS expected_loss,
    ROUND(SUM(s.churn_probability * c.monthly_charges * 12), 0)
                                                        AS expected_annual_revenue_at_risk
FROM customer_scores s
JOIN customers c USING (customer_id)
WHERE c.churn = 0
GROUP BY s.risk_level
ORDER BY avg_churn_probability_pct DESC;
