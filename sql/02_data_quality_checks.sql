-- ===========================================================================
-- 02_data_quality_checks.sql
-- Each query returns one row per named check. The pipeline runs them after the
-- schema is built and fails loudly if any check reports a non-zero violation,
-- so a silent data problem cannot reach the model.
-- ===========================================================================

-- CHECK: row_count
SELECT 'row_count' AS check_name,
       COUNT(*)    AS value,
       CASE WHEN COUNT(*) = 7043 THEN 0 ELSE 1 END AS violations
FROM customers;

-- CHECK: duplicate_customer_ids
SELECT 'duplicate_customer_ids' AS check_name,
       COUNT(*)                 AS value,
       COUNT(*)                 AS violations
FROM (SELECT customer_id FROM customers GROUP BY customer_id HAVING COUNT(*) > 1);

-- CHECK: null_monthly_charges
SELECT 'null_monthly_charges' AS check_name,
       COUNT(*)               AS value,
       COUNT(*)               AS violations
FROM customers WHERE monthly_charges IS NULL;

-- CHECK: negative_or_zero_charges
SELECT 'negative_or_zero_charges' AS check_name,
       COUNT(*)                   AS value,
       COUNT(*)                   AS violations
FROM customers WHERE monthly_charges <= 0 OR total_charges < 0;

-- CHECK: tenure_out_of_range
SELECT 'tenure_out_of_range' AS check_name,
       COUNT(*)              AS value,
       COUNT(*)              AS violations
FROM customers WHERE tenure < 0 OR tenure > 100;

-- CHECK: churn_not_binary
SELECT 'churn_not_binary' AS check_name,
       COUNT(*)           AS value,
       COUNT(*)           AS violations
FROM customers WHERE churn NOT IN (0, 1);

-- CHECK: zero_tenure_with_charges
-- A customer with tenure 0 cannot have been billed yet. Catches a bad cast.
SELECT 'zero_tenure_with_charges' AS check_name,
       COUNT(*)                   AS value,
       COUNT(*)                   AS violations
FROM customers WHERE tenure = 0 AND total_charges > 0;

-- CHECK: total_charges_implausible
-- total_charges should sit within a wide band of tenure * monthly_charges.
-- Prices change over time, so the band is deliberately generous; this is
-- looking for order-of-magnitude errors, not rounding noise.
-- One month of slack either side: the first version of this check flagged two
-- customers on tenure 2-3 whose ratio was 1.53-1.57. With so few bills, a single
-- plan change in month one explains that - they are real accounts, not errors.
SELECT 'total_charges_implausible' AS check_name,
       COUNT(*)                    AS value,
       COUNT(*)                    AS violations
FROM customers
WHERE tenure > 0
  AND (total_charges < (tenure - 1) * monthly_charges * 0.5
       OR total_charges > (tenure + 1) * monthly_charges * 1.5);

-- CHECK: orphan_internet_addons
-- An add-on cannot be active without the internet service it rides on.
SELECT 'orphan_internet_addons' AS check_name,
       COUNT(*)                 AS value,
       COUNT(*)                 AS violations
FROM customers
WHERE internet_service = 'No'
  AND (online_security = 'Yes' OR online_backup = 'Yes'
       OR tech_support = 'Yes' OR streaming_tv = 'Yes');
