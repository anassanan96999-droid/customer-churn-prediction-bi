-- ===========================================================================
-- 01_create_schema.sql
-- Builds the analytical layer for the churn project.
--
--   raw_customers   : the CSV landed verbatim, everything TEXT (a staging table)
--   customers       : typed, cleaned, de-duplicated analytical table
--
-- All cleaning that can be expressed declaratively is done here rather than in
-- pandas, so the transformation rules live next to the data.
-- ===========================================================================

DROP TABLE IF EXISTS customers;

CREATE TABLE customers AS
SELECT
    customerID                                          AS customer_id,
    gender,
    CAST(SeniorCitizen AS INTEGER)                      AS senior_citizen,
    Partner                                             AS partner,
    Dependents                                          AS dependents,
    CAST(tenure AS INTEGER)                             AS tenure,
    PhoneService                                        AS phone_service,

    -- "No phone service" / "No internet service" are not a third state: they
    -- are a "No" that carries the parent subscription's information, which the
    -- parent column already holds. Collapsing them avoids inventing a category.
    CASE WHEN MultipleLines    = 'No phone service'    THEN 'No' ELSE MultipleLines    END AS multiple_lines,
    InternetService                                     AS internet_service,
    CASE WHEN OnlineSecurity   = 'No internet service' THEN 'No' ELSE OnlineSecurity   END AS online_security,
    CASE WHEN OnlineBackup     = 'No internet service' THEN 'No' ELSE OnlineBackup     END AS online_backup,
    CASE WHEN DeviceProtection = 'No internet service' THEN 'No' ELSE DeviceProtection END AS device_protection,
    CASE WHEN TechSupport      = 'No internet service' THEN 'No' ELSE TechSupport      END AS tech_support,
    CASE WHEN StreamingTV      = 'No internet service' THEN 'No' ELSE StreamingTV      END AS streaming_tv,
    CASE WHEN StreamingMovies  = 'No internet service' THEN 'No' ELSE StreamingMovies  END AS streaming_movies,

    Contract                                            AS contract,
    PaperlessBilling                                    AS paperless_billing,
    PaymentMethod                                       AS payment_method,
    CAST(MonthlyCharges AS REAL)                        AS monthly_charges,

    -- 11 rows carry a blank TotalCharges. Every one of them has tenure = 0:
    -- they are customers who signed up but have not been billed yet, so the
    -- correct value is 0.0, not a median imputation.
    CASE
        WHEN TRIM(TotalCharges) = '' THEN 0.0
        ELSE CAST(TotalCharges AS REAL)
    END                                                 AS total_charges,

    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END           AS churn
FROM raw_customers
WHERE customerID IS NOT NULL
GROUP BY customerID;   -- collapses any duplicated customer record

CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_id ON customers (customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_churn    ON customers (churn);
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers (contract);
CREATE INDEX IF NOT EXISTS idx_customers_tenure   ON customers (tenure);

-- ---------------------------------------------------------------------------
-- Business features that are pure row-level arithmetic are derived in SQL.
-- Anything that needs to be *fitted* (scaling, encoding) stays in the Python
-- pipeline, where it can be fitted on the training fold only.
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS customer_features;

CREATE VIEW customer_features AS
SELECT
    c.*,

    -- Average bill across the whole life of the account. A customer who has
    -- not been billed yet (tenure 0) is, by definition, on their current bill.
    COALESCE(ROUND(c.total_charges / NULLIF(c.tenure, 0), 2),
             c.monthly_charges)                                AS avg_monthly_spend,

    -- > 1 means the customer is currently paying more than they historically
    -- did: a price rise, an up-sell, or a promo that has just expired.
    -- Stands in for "usage change %": the dataset has no usage history.
    COALESCE(ROUND(c.monthly_charges /
             NULLIF(c.total_charges / NULLIF(c.tenure, 0), 0), 3),
             1.0)                                              AS spend_trend_ratio,

    (CASE WHEN c.phone_service     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.multiple_lines    = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.internet_service <> 'No'  THEN 1 ELSE 0 END +
     CASE WHEN c.online_security   = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.online_backup     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.device_protection = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.tech_support      = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_tv      = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_movies  = 'Yes' THEN 1 ELSE 0 END)  AS service_count,

    (CASE WHEN c.streaming_tv     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_movies = 'Yes' THEN 1 ELSE 0 END)   AS streaming_count,

    -- Customers on a security/support bundle have a stickier relationship.
    CASE WHEN c.online_security = 'Yes' AND c.tech_support = 'Yes'
         THEN 1 ELSE 0 END                                     AS has_protection_bundle,

    CASE WHEN c.contract = 'Month-to-month' THEN 1 ELSE 0 END  AS is_month_to_month,
    CASE WHEN c.payment_method = 'Electronic check'
         THEN 1 ELSE 0 END                                     AS is_electronic_check,
    CASE WHEN c.payment_method LIKE '%automatic%'
         THEN 1 ELSE 0 END                                     AS is_auto_payment,
    CASE WHEN c.tenure <= 6 THEN 1 ELSE 0 END                  AS is_new_customer,
    CASE WHEN c.partner = 'Yes' OR c.dependents = 'Yes'
         THEN 1 ELSE 0 END                                     AS family_account,

    CASE
        WHEN c.tenure <= 6  THEN '0-6m'
        WHEN c.tenure <= 12 THEN '7-12m'
        WHEN c.tenure <= 24 THEN '1-2y'
        WHEN c.tenure <= 48 THEN '2-4y'
        ELSE                     '4y+'
    END                                                        AS tenure_bucket,

    -- Price per service: a customer paying a lot for very little is the
    -- classic "I'm not getting value for money" churn profile.
    ROUND(c.monthly_charges /
          NULLIF(CASE WHEN c.phone_service     = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.multiple_lines    = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.internet_service <> 'No'  THEN 1 ELSE 0 END +
                      CASE WHEN c.online_security   = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.online_backup     = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.device_protection = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.tech_support      = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.streaming_tv      = 'Yes' THEN 1 ELSE 0 END +
                      CASE WHEN c.streaming_movies  = 'Yes' THEN 1 ELSE 0 END, 0), 2)
                                                               AS charge_per_service,

    ROUND(c.monthly_charges * 12, 2)                           AS revenue_at_risk_annual
FROM customers c;
