import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import json
import os
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextDatasetGenerator:
    def __init__(self, db_config):
        """
        Inicializa o gerador de dataset

        Args:
            db_config (dict): Configurações do banco de dados
                - host: hostname do PostgreSQL
                - port: porta (default 5432)
                - database: nome do banco
                - username: usuário
                - password: senha
        """
        self.db_config = db_config
        self.engine = None
        self.connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    def connect(self):
        """Estabelece conexão com o banco"""
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
        """Cria a view de features para ML"""

        ml_view_sql = """
        CREATE OR REPLACE VIEW context_records_ml AS
        SELECT
          cr.id,
          cr.request_id,
          cr.created_at,

          -- RISK FEATURES
          (cr.risk_json->>'score')::float AS risk_score,
          COALESCE(cr.risk_json->>'level', 'unknown') AS risk_level,
          COALESCE((cr.risk_json->>'anomaly_score')::float, 0.0) AS risk_anomaly_score,
          COALESCE(cr.risk_json->>'model_version', 'unknown') AS risk_model_version,
          COALESCE((cr.risk_json->>'recent_incidents')::int, 0) AS risk_recent_incidents,
          -- policy_overrides como array JSON
          COALESCE(cr.risk_json->'policy_overrides', '[]'::jsonb) AS risk_policy_overrides,
          -- contagem de policy overrides
          COALESCE(jsonb_array_length(cr.risk_json->'policy_overrides'), 0) AS risk_policy_overrides_count,

          -- CONFIDENTIALITY FEATURES
          COALESCE((cr.confidentiality_json->>'score')::float, 0.0) AS conf_score,
          COALESCE(cr.confidentiality_json->>'classification', 'unknown') AS conf_classification,
          COALESCE(cr.confidentiality_json->'tags', '[]'::jsonb) AS conf_tags,
          COALESCE(jsonb_array_length(cr.confidentiality_json->'tags'), 0) AS conf_tags_count,
          COALESCE(cr.confidentiality_json->'detected_patterns', '[]'::jsonb) AS conf_detected_patterns,
          COALESCE(jsonb_array_length(cr.confidentiality_json->'detected_patterns'), 0) AS conf_patterns_count,
          COALESCE(cr.confidentiality_json->>'model_version','unknown') AS conf_model_version,

          -- SOURCE FEATURES
          COALESCE(cr.source_json->>'ip','0.0.0.0') AS src_ip,
          COALESCE(cr.source_json->>'geo','Unknown') AS src_geo,
          COALESCE(cr.source_json->>'user_id','') AS src_user_id,
          COALESCE(cr.source_json->>'device_id','') AS src_device_id,
          COALESCE(cr.source_json->>'user_agent','unknown') AS src_user_agent,
          COALESCE(cr.source_json->>'os_version','unknown') AS src_os_version,
          COALESCE(cr.source_json->>'device_type','unknown') AS src_device_type,
          COALESCE(cr.source_json->>'mfa_status','unknown') AS src_mfa_status,
          COALESCE(cr.source_json->>'security_status','unknown') AS src_security_status,

          -- DESTINATION FEATURES
          COALESCE(cr.destination_json->>'ip','0.0.0.0') AS dst_ip,
          COALESCE(cr.destination_json->>'service_id','') AS dst_service_id,
          COALESCE(cr.destination_json->>'service_type','unknown') AS dst_service_type,
          COALESCE(cr.destination_json->>'security_policy','unknown') AS dst_security_policy,
          COALESCE(cr.destination_json->>'security_status','unknown') AS dst_security_status,
          COALESCE(cr.destination_json->>'os_version','unknown') AS dst_os_version,
          COALESCE(cr.destination_json->'allowed_protocols', '[]'::jsonb) AS dst_allowed_protocols,
          -- verifica se TLS1.3 está permitido
          COALESCE((cr.destination_json->'allowed_protocols') ? 'TLS1.3', false) AS dst_tls13_allowed,
          -- contagem de protocolos permitidos
          COALESCE(jsonb_array_length(cr.destination_json->'allowed_protocols'), 0) AS dst_protocols_count,

          -- HEADERS FEATURES
          COALESCE(cr.headers_json, '{}'::jsonb) AS headers_json,
          -- contagem de headers
          (
            SELECT COUNT(*) 
            FROM jsonb_object_keys(COALESCE(cr.headers_json,'{}'::jsonb))
          ) AS headers_count,

          -- TIME FEATURES
          EXTRACT(HOUR FROM cr.created_at) AS hour_of_day,
          EXTRACT(DOW FROM cr.created_at) AS day_of_week,
          EXTRACT(MONTH FROM cr.created_at) AS month,
          EXTRACT(YEAR FROM cr.created_at) AS year,

          -- DERIVED FEATURES
          -- IP privado (192.168.*, 10.*, 172.16-31.*)
          CASE 
            WHEN COALESCE(cr.source_json->>'ip','') ~ '^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)' THEN true
            ELSE false
          END AS src_ip_private,

          -- Normalização do status MFA
          CASE lower(COALESCE(cr.source_json->>'mfa_status','unknown'))
            WHEN 'enabled' THEN 'enabled'
            WHEN 'disabled' THEN 'disabled'
            ELSE 'unknown'
          END AS src_mfa_status_norm,

          -- Score combinado (risk + confidentiality)
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
        """Cria a view com labels para treinamento"""

        labels_view_sql = """
        CREATE OR REPLACE VIEW context_records_labels AS
        SELECT
          *,
          -- SECURITY LEVEL LABEL
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

          -- ENCRYPTION SCRIPT LABEL
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

          -- PRIORITY LABEL (para ordenação de processamento)
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
        """Retorna informações sobre o dataset"""

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
                # Informações gerais
                info_df = pd.read_sql(info_sql, conn)
                logger.info("Informações do dataset:")
                logger.info(f"Total de registros: {info_df.iloc[0]['total_records']}")
                logger.info(f"Níveis de segurança únicos: {info_df.iloc[0]['security_levels']}")
                logger.info(f"Scripts de criptografia únicos: {info_df.iloc[0]['encryption_scripts']}")
                logger.info(f"Período: {info_df.iloc[0]['oldest_record']} até {info_df.iloc[0]['newest_record']}")

                # Distribuição das classes
                dist_df = pd.read_sql(distribution_sql, conn)
                logger.info("\nDistribuição das classes:")
                print(dist_df.to_string(index=False))

                return info_df, dist_df

        except Exception as e:
            logger.error(f"Erro ao obter informações do dataset: {e}")
            return None, None

    def export_dataset(self, output_file='context_dataset.csv', sample_size=None):
        """
        Exporta o dataset para CSV

        Args:
            output_file (str): Nome do arquivo de saída
            sample_size (int): Tamanho da amostra (None para todos os registros)
        """

        # Query principal para exportação
        export_sql = """
        SELECT
            id, request_id, created_at,

            -- Risk features
            risk_score, risk_level, risk_anomaly_score, risk_recent_incidents,
            risk_policy_overrides_count,

            -- Confidentiality features  
            conf_score, conf_classification, conf_tags_count, conf_patterns_count,

            -- Source features
            src_ip_private, src_geo, src_mfa_status_norm, src_device_type, src_security_status,

            -- Destination features
            dst_service_type, dst_security_policy, dst_tls13_allowed, dst_protocols_count,

            -- Headers and time features
            headers_count, hour_of_day, day_of_week, month,

            -- Derived features
            combined_score,

            -- Labels (targets)
            security_level_label, encryption_script_label, processing_priority_label

        FROM context_records_labels
        ORDER BY created_at DESC
        """

        if sample_size:
            export_sql += f" LIMIT {sample_size}"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(export_sql, conn)

                # Salva o CSV
                df.to_csv(output_file, index=False)
                logger.info(f"Dataset exportado para '{output_file}' com {len(df)} registros")

                # Estatísticas básicas
                logger.info("\nEstatísticas do dataset exportado:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Colunas: {list(df.columns)}")

                # Distribuição das classes target
                logger.info("\nDistribuição Security Level:")
                print(df['security_level_label'].value_counts())

                logger.info("\nDistribuição Encryption Script:")
                print(df['encryption_script_label'].value_counts())

                return df

        except Exception as e:
            logger.error(f"Erro ao exportar dataset: {e}")
            return None

    def create_feature_engineering_dataset(self, output_file='context_dataset_engineered.csv'):
        """
        Cria um dataset com feature engineering mais avançado
        """

        advanced_sql = """
        WITH feature_engineering AS (
          SELECT *,
            -- One-hot (já existiam)
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
        
            -- Cíclicas (já existiam)
            SIN(2 * PI() * hour_of_day / 24.0) as hour_sin,
            COS(2 * PI() * hour_of_day / 24.0) as hour_cos,
            SIN(2 * PI() * day_of_week / 7.0) as day_sin,
            COS(2 * PI() * day_of_week / 7.0) as day_cos,
        
            -- Interações/normalizações (já existiam)
            risk_score * conf_score as risk_conf_interaction,
            CASE WHEN src_ip_private AND dst_tls13_allowed THEN 1 ELSE 0 END as private_secure_combo,
            (risk_score - 0.5) / 0.5 as risk_score_normalized,
            (conf_score - 0.5) / 0.5 as conf_score_normalized,
        
            -- Fallback do request_id
            COALESCE(request_id, 'req-' || id::text) AS request_id_resolved
        
          FROM context_records_labels
        )
        SELECT
          id,
          request_id_resolved,         -- novo
          created_at,
        
          -- Numéricas originais
          risk_score, risk_anomaly_score, risk_recent_incidents,
          conf_score,
          headers_count, combined_score,
        
          -- Contagens/flags adicionais
          risk_policy_overrides_count, -- novo
          conf_tags_count, conf_patterns_count,
          dst_protocols_count,
          src_ip_private, dst_tls13_allowed,
        
          -- Categóricas cruas para debug/interpretabilidade
          risk_level,                  -- novo
          conf_classification,         -- novo
          src_geo,                     -- novo
          src_device_type,             -- novo
          dst_service_type,            -- novo
          dst_security_policy,         -- novo
          src_mfa_status_norm,         -- novo
        
          -- Tempo cru + cíclico
          hour_of_day, day_of_week, month, year,  -- novos (os crus)
          hour_sin, hour_cos, day_sin, day_cos,
        
          -- One-hot
          risk_level_low, risk_level_medium, risk_level_high, risk_level_critical,
          conf_public, conf_internal, conf_confidential, conf_restricted,
          src_mfa_enabled, src_mfa_disabled,
          dst_policy_low, dst_policy_medium, dst_policy_high,
        
          -- Interações/normalizações
          risk_conf_interaction, private_secure_combo,
          risk_score_normalized, conf_score_normalized,
        
          -- Targets
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
    """Função principal"""

    # Configuração do banco - AJUSTE AQUI SUAS CREDENCIAIS
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'context_db',
        'username': 'postgres',
        'password': 'postgres'  # MUDE PARA SUA SENHA
    }

    # Inicializa o gerador
    generator = ContextDatasetGenerator(db_config)

    # Conecta ao banco
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

    # Obtém informações do dataset
    logger.info("Analisando dataset...")
    info_df, dist_df = generator.get_dataset_info()

    if info_df is not None:
        # Exporta dataset básico
        logger.info("Exportando dataset básico...")
        basic_df = generator.export_dataset('context_dataset_basic.csv')

        # Exporta dataset com feature engineering
        logger.info("Exportando dataset com feature engineering...")
        advanced_df = generator.create_feature_engineering_dataset('context_dataset_advanced.csv')

        # Exporta uma amostra pequena para testes
        logger.info("Exportando amostra para testes...")
        sample_df = generator.export_dataset('context_dataset_sample.csv', sample_size=100)

        logger.info("✅ Processo concluído com sucesso!")
        logger.info("Arquivos gerados:")
        logger.info("- context_dataset_basic.csv (dataset básico)")
        logger.info("- context_dataset_advanced.csv (com feature engineering)")
        logger.info("- context_dataset_sample.csv (amostra de 100 registros)")

    else:
        logger.error("Não foi possível analisar o dataset. Verifique se há dados na tabela context_records.")


if __name__ == "__main__":
    main()