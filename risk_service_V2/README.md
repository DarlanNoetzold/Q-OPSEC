


1. Pesquisar técnicas de criação de datasets
   2. 
2. Definir todas as propriedades que serão analisadas
3. Validar e ter o diferencial nessas propriedades
4. Definir LLM
5. Local e forma de extração de Dados
6. Automatizar a criação do dataset
7. Como essas informações serão analisadas pelo LLM
8. Pesquisar sobre Níveis de Segurança
9. Usar modelo do Estado da arte de classificação para definir Nível de segurança




1.1. Event identification
event_id
event_type – e.g. transaction, login, password_change, …
event_source – e.g. web_app, mobile_app, api_partner, batch_job
timestamp_utc
timestamp_local
timezone
1.2. User / account
user_id (anonymized)
account_id
user_type – individual, business, admin, system_user, …
user_segment – retail, corporate, vip, …
user_risk_class – existing risk level, if you have it
account_age_days
account_creation_date
sensitive_data_change_last_7d – 0/1 (email/phone/address changed recently)
registered_devices_count
active_devices_last_30d
1.3. Temporal / behavioral features
hour_of_day – 0–23
day_of_week – 0–6
is_weekend – 0/1
is_local_holiday – 0/1
seconds_since_last_login
seconds_since_last_transaction
transactions_last_1h
transactions_last_24h
transactions_last_7d
transactions_last_30d
amount_sum_last_24h
amount_sum_last_7d
amount_sum_last_30d
amount_mean_last_30d
amount_std_last_30d
logins_last_24h
login_failures_last_24h
password_resets_last_30d
1.4. Transaction / business content
amount
currency
transaction_type – pix, wire, card, internal_transfer, …
transaction_subtype – online_purchase, cash_withdrawal, bill_payment, …
channel – web, mobile_android, mobile_ios, atm, call_center, api_partner
destination_type – same_bank_account, other_bank_account, wallet, …
destination_id (anonymized)
destination_bank
destination_country
destination_segment
is_new_recipient – 0/1
is_first_transaction_to_recipient – 0/1
transactions_to_recipient_last_7d
amount_sum_to_recipient_last_7d
amount_increase_vs_30d_mean – ratio or difference
amount_percentile_for_user_30d
1.5. Location / network
ip_address (hash/anonymized if needed)
ip_version – 4/6
origin_country
origin_region
origin_city
origin_asn
origin_isp
geo_latitude (rounded)
geo_longitude (rounded)
registered_country
registered_region
distance_km_from_last_location
distance_km_from_registered_address
country_change_since_last_session – 0/1
city_change_since_last_session – 0/1
ip_on_blacklist – 0/1
ip_reputation_score – 0–100
distinct_ips_last_24h
distinct_countries_last_7d
1.6. Device / environment
device_id / device_fingerprint
device_type – mobile, desktop, tablet, iot, …
device_os – Android, iOS, Windows, Linux, macOS, …
device_os_version
device_model
browser_name – Chrome, Firefox, …
browser_version
user_agent (or hashed)
is_new_device – 0/1
distinct_devices_last_7d
new_devices_last_30d
is_jailbroken_or_rooted – 0/1
is_emulator_detected – 0/1
is_vpn_detected – 0/1
is_proxy_detected – 0/1
is_tor_detected – 0/1
1.7. Authentication / authorization / security tech & versions
auth_method – password, oauth, certificate, hardware_token, biometrics, …
auth_context_level – low, medium, high
mfa_used – 0/1
mfa_type – sms, app_token, push, biometrics, email, fido2, …
mfa_success – 0/1
mfa_failures_last_24h
session_id
session_age_seconds
session_risk_score
user_role – admin, operator, end_user, auditor, …
Security technologies + versions:

tls_version – TLS_1_0, TLS_1_2, TLS_1_3, …
tls_cipher_suite
tls_weak_cipher – 0/1
token_type – JWT, opaque, SAML, …
token_sign_algorithm – HS256, RS256, ES256, …
token_issuer
token_audience
token_expiry_seconds
waf_vendor
waf_version
waf_rule_matched – rule id/name, if any
ids_ips_vendor
ids_ips_version
ids_alert_severity – low, medium, high, critical
ids_alert_type – sql_injection, xss, brute_force, …
client_av_vendor
client_av_version
device_security_patch_level
os_security_features_enabled – e.g. bitmask/string (secure_boot, disk_encryption, …)
app_version
app_build_number
api_version
crypto_library_name – OpenSSL, BoringSSL, …
crypto_library_version
password_hash_algorithm – bcrypt, scrypt, argon2, pbkdf2, legacy_md5, …
password_policy_version
1.8. Fraud / abuse / historical risk
user_has_fraud_history – 0/1
user_fraud_incidents_count
user_chargebacks_last_12m
user_on_blacklist – 0/1
device_on_blacklist – 0/1
card_on_blacklist – 0/1 (if applicable)
risk_alerts_last_30d
user_global_risk_score
destination_on_blacklist – 0/1
1.9. Text / message content (if applicable)
message_text (or message_text_hash if you can’t share raw text)
message_length_chars
message_links_count
message_attachments_count
message_attachment_types – e.g. pdf;zip;exe
message_contains_sensitive_keywords – 0/1
message_language – en, pt, es, …
Optional pre-LLM text features:

message_embedding_dimN (if you release vector embeddings)
message_toxicity_score
message_phishing_keyword_score
1.10. LLM-derived features
Numeric scores:

llm_risk_score – 0–1
llm_risk_level – e.g. low, medium, high, critical
llm_phishing_score – 0–1
llm_social_engineering_score – 0–1
llm_urgency_score – 0–1
llm_sensitivity_request_score – 0–1 (asks for sensitive data)
Categorical classifications:

llm_risk_category – financial_fraud, phishing, social_engineering, unauthorized_access, benign, other
llm_fraud_pattern – account_takeover, card_not_present, money_mule, …
llm_intent – pay_bill, transfer_to_self, transfer_to_third_party, test_transaction, …
Boolean signals:

llm_detected_social_engineering_language – 0/1
llm_detected_request_for_personal_data – 0/1
llm_detected_threat_or_coercion – 0/1
llm_detected_suspicious_link – 0/1
Explainability / metadata:

llm_short_explanation
llm_risk_tags – e.g. ["new_recipient", "high_amount", "urgent_message"]
llm_model_name
llm_model_version
llm_prompt_version
llm_inference_timestamp
1.11. Labels / ground truth
label_fraud – 0/1
label_risk_level – 0=low, 1=medium, 2=high, 3=critical
label_fraud_type – if known
label_source – confirmed_incident, rule_engine, anomaly_model, customer_dispute, …
label_confidence – 0–1
label_version – version of labeling strategy