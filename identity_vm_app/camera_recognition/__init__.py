"""Luồng nhận diện theo camera (bật/tắt qua /ivm/cameras/{id}/analyze) — pattern VisionMaster."""

from identity_vm_app.camera_recognition.hub import (
    RecognitionHub,
    ensure_recognition_hub_started,
    get_recognition_hub,
    shutdown_recognition_hub,
    start_recognition_hub,
)

__all__ = [
    "RecognitionHub",
    "get_recognition_hub",
    "start_recognition_hub",
    "ensure_recognition_hub_started",
    "shutdown_recognition_hub",
]
