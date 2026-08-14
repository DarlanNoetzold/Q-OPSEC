from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings
from app.models.database import PredictionRecord, ModelRecord, MetricRecord
import structlog

logger = structlog.get_logger()


class DatabaseService:
    

    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database = None

    async def connect(self):
        
        try:
            print(f"Connecting to MongoDB at {settings.mongodb_url}...")
            self.client = AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
            self.database = self.client[settings.mongodb_database]

            try:
                await self.client.admin.command('ping')
            except Exception as e:
                print(f"CRITICAL: MongoDB not reachable: {e}. Proceeding without DB...")
                return

            await init_beanie(
                database=self.database,
                document_models=[
                    PredictionRecord,
                    ModelRecord,
                    MetricRecord
                ]
            )

            await self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")

        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise

    async def disconnect(self):
        
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def health_check(self) -> bool:
        
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False


db_service = DatabaseService()


async def get_database():
    
    return db_service