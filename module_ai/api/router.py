"""Combined AI HTTP routers for host mount."""

from fastapi import APIRouter

from module_ai.api.face_routes import router as face_router


def build_ai_router() -> APIRouter:
    root = APIRouter()
    root.include_router(face_router)
    try:
        from module_ai.api.people_routes import router as people_router
        from module_ai.api.infer_routes import router as infer_router
        from module_ai.api.video_analyze_routes import router as video_analyze_router
        from module_ai.api.vm_reports_routes import router as vm_reports_router
        from module_ai.api.weapon_alerts_routes import router as weapon_alerts_router

        root.include_router(people_router)
        root.include_router(infer_router)
        root.include_router(video_analyze_router)
        root.include_router(vm_reports_router)
        root.include_router(weapon_alerts_router)
    except ImportError:
        pass
    return root
