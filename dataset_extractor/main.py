import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextDatasetGenerator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = None
        self.connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    def connect(self):
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Conexão com PostgreSQL estabelecida com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar com PostgreSQL: {e}")
            return False

    def drop_views(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DROP VIEW IF EXISTS context_records_labels"))
                conn.execute(text("DROP VIEW IF EXISTS context_records_ml"))
                conn.commit()
            logger.info("Views antigas dropadas (labels, ml)")
            return True
        except Exception as e:
            logger.error(f"Erro ao dropar views: {e}")
            return False

    def create_ml_view(self):

        ml_view_sql = """
        CREATE OR REPLACE VIEW context_records_ml AS
        SELECT
          cr.id,
          cr.request_id,
          cr.created_at,

          (cr.risk_json->>'score')::float AS risk_score,
          COALESCE(cr.risk_json->>'level', 'unknown') AS risk_level,
          COALESCE((cr.risk_json->>'anomaly_score')::float, 0.0) AS risk_anomaly_score,
          COALESCE(cr.risk_json->>'model_version', 'unknown') AS risk_model_version,
          COALESCE((cr.risk_json->>'recent_incidents')::int, 0) AS risk_recent_incidents,
          COALESCE(cr.risk_json->'policy_overrides', '[]'::jsonb) AS risk_policy_overrides,
          COALESCE(jsonb_array_length(cr.risk_json->'policy_overrides'), 0) AS risk_policy_overrides_count,

          COALESCE((cr.confidentiality_json->>'score')::float, 0.0) AS conf_score,
          COALESCE(cr.confidentiality_json->>'classification', 'unknown') AS conf_classification,
          COALESCE(cr.confidentiality_json->'tags', '[]'::jsonb) AS conf_tags,
          COALESCE(jsonb_array_length(cr.confidentiality_json->'tags'), 0) AS conf_tags_count,
          COALESCE(cr.confidentiality_json->'detected_patterns', '[]'::jsonb) AS conf_detected_patterns,
          COALESCE(jsonb_array_length(cr.confidentiality_json->'detected_patterns'), 0) AS conf_patterns_count,
          COALESCE(cr.confidentiality_json->>'model_version','unknown') AS conf_model_version,

          COALESCE(cr.source_json->>'ip','0.0.0.0') AS src_ip,
          COALESCE(cr.source_json->>'geo','Unknown') AS src_geo,
          COALESCE(cr.source_json->>'user_id','') AS src_user_id,
          COALESCE(cr.source_json->>'device_id','') AS src_device_id,
          COALESCE(cr.source_json->>'user_agent','unknown') AS src_user_agent,
          COALESCE(cr.source_json->>'os_version','unknown') AS src_os_version,
          COALESCE(cr.source_json->>'device_type','unknown') AS src_device_type,
          COALESCE(cr.source_json->>'mfa_status','unknown') AS src_mfa_status,
          COALESCE(cr.source_json->>'security_status','unknown') AS src_security_status,

          COALESCE(cr.destination_json->>'ip','0.0.0.0') AS dst_ip,
          COALESCE(cr.destination_json->>'service_id','') AS dst_service_id,
          COALESCE(cr.destination_json->>'service_type','unknown') AS dst_service_type,
          COALESCE(cr.destination_json->>'security_policy','unknown') AS dst_security_policy,
          COALESCE(cr.destination_json->>'security_status','unknown') AS dst_security_status,
          COALESCE(cr.destination_json->>'os_version','unknown') AS dst_os_version,
          COALESCE(cr.destination_json->'allowed_protocols', '[]'::jsonb) AS dst_allowed_protocols,
          COALESCE((cr.destination_json->'allowed_protocols') ? 'TLS1.3', false) AS dst_tls13_allowed,
          COALESCE(jsonb_array_length(cr.destination_json->'allowed_protocols'), 0) AS dst_protocols_count,

          COALESCE(cr.headers_json, '{}'::jsonb) AS headers_json,
          (
            SELECT COUNT(*) 
            FROM jsonb_object_keys(COALESCE(cr.headers_json,'{}'::jsonb))
          ) AS headers_count,

          EXTRACT(HOUR FROM cr.created_at) AS hour_of_day,
          EXTRACT(DOW FROM cr.created_at) AS day_of_week,
          EXTRACT(MONTH FROM cr.created_at) AS month,
          EXTRACT(YEAR FROM cr.created_at) AS year,

          CASE 
            WHEN COALESCE(cr.source_json->>'ip','') ~ '^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)' THEN true
            ELSE false
          END AS src_ip_private,

          CASE lower(COALESCE(cr.source_json->>'mfa_status','unknown'))
            WHEN 'enabled' THEN 'enabled'
            WHEN 'disabled' THEN 'disabled'
            ELSE 'unknown'
          END AS src_mfa_status_norm,

          COALESCE((cr.risk_json->>'score')::float, 0.0) * 0.6 + 
          COALESCE((cr.confidentiality_json->>'score')::float, 0.0) * 0.4 AS combined_score

        FROM context_records cr;
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(ml_view_sql))
                conn.commit()
            logger.info("View 'context_records_ml' criada com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar view ML: {e}")
            return False

    def create_labels_view(self):

        labels_view_sql = """
        CREATE OR REPLACE VIEW context_records_labels AS
        SELECT
          *,
          CASE
            WHEN lower(dst_security_policy) = 'high'
              OR lower(conf_classification) IN ('confidential','restricted','secret')
              OR lower(risk_level) IN ('high','critical')
              OR risk_score > 0.8
              OR combined_score > 0.8
            THEN 'critical'
            WHEN lower(risk_level) = 'medium'
              OR lower(conf_classification) = 'internal'
              OR risk_score BETWEEN 0.4 AND 0.8
              OR combined_score BETWEEN 0.4 AND 0.8
              OR risk_recent_incidents > 0
            THEN 'medium'
            ELSE 'low'
          END AS security_level_label,

          CASE
            WHEN (
              lower(dst_security_policy) = 'high'
              OR lower(conf_classification) IN ('confidential','restricted','secret')
              OR lower(risk_level) IN ('high','critical')
              OR risk_score > 0.8
              OR combined_score > 0.8
            )
              THEN 'mtls_aes256_gcm_x25519'
            WHEN (
              lower(risk_level) = 'medium'
              OR lower(conf_classification) = 'internal'
              OR risk_score BETWEEN 0.4 AND 0.8
              OR combined_score BETWEEN 0.4 AND 0.8
              OR risk_recent_incidents > 0
            )
              THEN CASE 
                WHEN dst_tls13_allowed THEN 'aes256_gcm_tls13' 
                ELSE 'chacha20_poly1305' 
              END
            ELSE CASE
              WHEN dst_tls13_allowed THEN 'aes128_gcm_tls13'
              ELSE 'aes128_gcm'
            END
          END AS encryption_script_label,

          CASE
            WHEN lower(dst_security_policy) = 'high'
              OR lower(conf_classification) IN ('confidential','restricted','secret')
              OR lower(risk_level) IN ('high','critical')
            THEN 'high_priority'
            WHEN lower(risk_level) = 'medium'
              OR lower(conf_classification) = 'internal'
            THEN 'medium_priority'
            ELSE 'low_priority'
          END AS processing_priority_label

        FROM context_records_ml;
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(labels_view_sql))
                conn.commit()
            logger.info("View 'context_records_labels' criada com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar view de labels: {e}")
            return False

    def get_dataset_info(self):

        info_sql = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT security_level_label) as security_levels,
            COUNT(DISTINCT encryption_script_label) as encryption_scripts,
            MIN(created_at) as oldest_record,
            MAX(created_at) as newest_record
        FROM context_records_labels;
        """

        distribution_sql = """
        SELECT 
            security_level_label,
            encryption_script_label,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM context_records_labels
        GROUP BY security_level_label, encryption_script_label
        ORDER BY count DESC;
        """

        try:
            with self.engine.connect() as conn:
                info_df = pd.read_sql(info_sql, conn)
                logger.info("Informações do dataset:")
                logger.info(f"Total de registros: {info_df.iloc[0]['total_records']}")
                logger.info(f"Níveis de segurança únicos: {info_df.iloc[0]['security_levels']}")
                logger.info(f"Scripts de criptografia únicos: {info_df.iloc[0]['encryption_scripts']}")
                logger.info(f"Período: {info_df.iloc[0]['oldest_record']} até {info_df.iloc[0]['newest_record']}")

                dist_df = pd.read_sql(distribution_sql, conn)
                logger.info("\nDistribuição das classes:")
                print(dist_df.to_string(index=False))

                return info_df, dist_df

        except Exception as e:
            logger.error(f"Erro ao obter informações do dataset: {e}")
            return None, None

    def export_dataset(self, output_file='context_dataset.csv', sample_size=None):

        export_sql = """
        SELECT
            id, request_id, created_at,

            risk_score, risk_level, risk_anomaly_score, risk_recent_incidents,
            risk_policy_overrides_count,

            conf_score, conf_classification, conf_tags_count, conf_patterns_count,

            src_ip_private, src_geo, src_mfa_status_norm, src_device_type, src_security_status,

            dst_service_type, dst_security_policy, dst_tls13_allowed, dst_protocols_count,

            headers_count, hour_of_day, day_of_week, month,

            combined_score,

            security_level_label, encryption_script_label, processing_priority_label

        FROM context_records_labels
        ORDER BY created_at DESC
        """

        if sample_size:
            export_sql += f" LIMIT {sample_size}"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(export_sql, conn)

                df.to_csv(output_file, index=False)
                logger.info(f"Dataset exportado para '{output_file}' com {len(df)} registros")

                logger.info("\nEstatísticas do dataset exportado:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Colunas: {list(df.columns)}")

                logger.info("\nDistribuição Security Level:")
                print(df['security_level_label'].value_counts())

                logger.info("\nDistribuição Encryption Script:")
                print(df['encryption_script_label'].value_counts())

                return df

        except Exception as e:
            logger.error(f"Erro ao exportar dataset: {e}")
            return None

    def create_feature_engineering_dataset(self, output_file='context_dataset_engineered.csv'):

        advanced_sql = """
        WITH feature_engineering AS (
          SELECT *,
            CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END as risk_level_low,
            CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END as risk_level_medium,
            CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END as risk_level_high,
            CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END as risk_level_critical,
        
            CASE WHEN conf_classification = 'public' THEN 1 ELSE 0 END as conf_public,
            CASE WHEN conf_classification = 'internal' THEN 1 ELSE 0 END as conf_internal,
            CASE WHEN conf_classification = 'confidential' THEN 1 ELSE 0 END as conf_confidential,
            CASE WHEN conf_classification = 'restricted' THEN 1 ELSE 0 END as conf_restricted,
        
            CASE WHEN src_mfa_status_norm = 'enabled' THEN 1 ELSE 0 END as src_mfa_enabled,
            CASE WHEN src_mfa_status_norm = 'disabled' THEN 1 ELSE 0 END as src_mfa_disabled,
        
            CASE WHEN dst_security_policy = 'low' THEN 1 ELSE 0 END as dst_policy_low,
            CASE WHEN dst_security_policy = 'medium' THEN 1 ELSE 0 END as dst_policy_medium,
            CASE WHEN dst_security_policy = 'high' THEN 1 ELSE 0 END as dst_policy_high,
        
            SIN(2 * PI() * hour_of_day / 24.0) as hour_sin,
            COS(2 * PI() * hour_of_day / 24.0) as hour_cos,
            SIN(2 * PI() * day_of_week / 7.0) as day_sin,
            COS(2 * PI() * day_of_week / 7.0) as day_cos,
        
            risk_score * conf_score as risk_conf_interaction,
            CASE WHEN src_ip_private AND dst_tls13_allowed THEN 1 ELSE 0 END as private_secure_combo,
            (risk_score - 0.5) / 0.5 as risk_score_normalized,
            (conf_score - 0.5) / 0.5 as conf_score_normalized,
        
            COALESCE(request_id, 'req-' || id::text) AS request_id_resolved
        
          FROM context_records_labels
        )
        SELECT
          id,
          request_id_resolved,
          created_at,
        
          risk_score, risk_anomaly_score, risk_recent_incidents,
          conf_score,
          headers_count, combined_score,
        
          risk_policy_overrides_count,
          conf_tags_count, conf_patterns_count,
          dst_protocols_count,
          src_ip_private, dst_tls13_allowed,
        
          risk_level,
          conf_classification,
          src_geo,
          src_device_type,
          dst_service_type,
          dst_security_policy,
          src_mfa_status_norm,
        
          hour_of_day, day_of_week, month, year,
          hour_sin, hour_cos, day_sin, day_cos,
        
          risk_level_low, risk_level_medium, risk_level_high, risk_level_critical,
          conf_public, conf_internal, conf_confidential, conf_restricted,
          src_mfa_enabled, src_mfa_disabled,
          dst_policy_low, dst_policy_medium, dst_policy_high,
        
          risk_conf_interaction, private_secure_combo,
          risk_score_normalized, conf_score_normalized,
        
          security_level_label, encryption_script_label, processing_priority_label
        
        FROM feature_engineering
        ORDER BY created_at DESC;
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(advanced_sql, conn)
                df.to_csv(output_file, index=False)
                logger.info(f"Dataset com feature engineering exportado para '{output_file}' com {len(df)} registros")
                logger.info(f"Features: {len(df.columns)} colunas")
                return df

        except Exception as e:
            logger.error(f"Erro ao criar dataset com feature engineering: {e}")
            return None


def main():

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'context_db',
        'username': 'postgres',
        'password': 'postgres'
    }

    generator = ContextDatasetGenerator(db_config)

    if not generator.connect():
        logger.error("Falha na conexão. Verifique as credenciais do banco.")
        return

    logger.info("Removendo views...")
    if not generator.drop_views():
        return

    logger.info("Criando views...")
    if not generator.create_ml_view():
        return

    if not generator.create_labels_view():
        return

    logger.info("Analisando dataset...")
    info_df, dist_df = generator.get_dataset_info()

    if info_df is not None:
        logger.info("Exportando dataset básico...")
        basic_df = generator.export_dataset(r'C:\Projetos\Q-OPSEC\classify_scheduler\datasets\v1\context_dataset_basic.csv')

        logger.info("Exportando dataset com feature engineering...")
        advanced_df = generator.create_feature_engineering_dataset(r'C:\Projetos\Q-OPSEC\classify_scheduler\datasets\v1\context_dataset_advanced.csv')

        logger.info("Exportando amostra para testes...")
        sample_df = generator.export_dataset(r'C:\Projetos\Q-OPSEC\classify_scheduler\datasets\v1\context_dataset_sample.csv', sample_size=1000)

        logger.info("✅ Processo concluído com sucesso!")
        logger.info("Arquivos gerados:")
        logger.info("- context_dataset_basic.csv (dataset básico)")
        logger.info("- context_dataset_advanced.csv (com feature engineering)")
        logger.info("- context_dataset_sample.csv (amostra de 100 registros)")

    else:
        logger.error("Não foi possível analisar o dataset. Verifique se há dados na tabela context_records.")


if __name__ == "__main__":
    main()