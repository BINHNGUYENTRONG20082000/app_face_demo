"""Legacy entry: aggregate AI + host IVM routers."""

from fastapi import APIRouter

from identity_vm_app.api.routes_host import router as host_router
from module_ai.api.face_routes import router as face_router

router = APIRouter()
router.include_router(face_router)
router.include_router(host_router)
