from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from identity_vm_app.api.camera_analyze_reports_routes import router as camera_analyze_reports_router
from identity_vm_app.api.preview_native_routes import router as preview_native_router
from identity_vm_app.api.preview_routes import router as preview_router
from identity_vm_app.api.routes import router as ivm_router
from module_ai.api.infer_routes import router as infer_router
from module_ai.api.people_routes import router as people_router
from module_ai.api.video_analyze_routes import router as video_analyze_router
from module_ai.api.vm_reports_routes import router as vm_reports_router
from module_ai.api.weapon_alerts_routes import router as weapon_alerts_router
from module_ai.camera.hub import shutdown_recognition_hub
from identity_vm_app.lifecycle import shutdown, startup


def create_app() -> FastAPI:
    app = FastAPI(
        title="Identity VM (InsightFace)",
        version="0.2.0",
        description="Nhận diện InsightFace-only, archive RTSP, báo cáo camera, export cut on-demand, gallery.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def on_start():
        startup()

    @app.on_event("shutdown")
    async def on_shutdown():
        shutdown_recognition_hub()
        shutdown()

    app.include_router(ivm_router)
    app.include_router(infer_router)
    app.include_router(preview_router)
    app.include_router(preview_native_router)
    app.include_router(people_router)
    app.include_router(video_analyze_router)
    app.include_router(vm_reports_router)
    app.include_router(camera_analyze_reports_router)
    app.include_router(weapon_alerts_router)
    return app
