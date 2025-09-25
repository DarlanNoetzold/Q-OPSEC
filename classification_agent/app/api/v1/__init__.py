"""
API v1 package.
"""
from fastapi import APIRouter
from .endpoints import router as endpoints_router

# Create the main v1 router
router = APIRouter()
router.include_router(endpoints_router)

__all__ = ["router"]