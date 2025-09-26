# app/services/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings
from app.models.database import PredictionRecord, ModelRecord, MetricRecord
import structlog

logger = structlog.get_logger()


class DatabaseService:
    """Serviço para gerenciar conexões com MongoDB"""

    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database = None

    async def connect(self):
        """Conecta ao MongoDB"""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.database = self.client[settings.mongodb_database]

            # Inicializa Beanie com os modelos
            await init_beanie(
                database=self.database,
                document_models=[
                    PredictionRecord,
                    ModelRecord,
                    MetricRecord
                ]
            )

            # Testa a conexão
            await self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")

        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise

    async def disconnect(self):
        """Desconecta do MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def health_check(self) -> bool:
        """Verifica se a conexão com o banco está saudável"""
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False


# Instância global do serviço de banco
db_service = DatabaseService()


async def get_database():
    """Dependency para obter a instância do banco"""
    return db_service