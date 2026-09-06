from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys
import asyncio

from cryptography.fernet import Fernet
import httpx


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


temp_db_handle, temp_db_name = tempfile.mkstemp(prefix="woundcare_professor_preview_", suffix=".db")
os.close(temp_db_handle)
temp_db = Path(temp_db_name)
os.environ.update(
    {
        "WOUNDCARE_DB_PATH": str(temp_db),
        "WOUNDCARE_FERNET_KEY": Fernet.generate_key().decode(),
        "WOUNDCARE_ENV": "development",
        "WOUNDCARE_PROFESSOR_PREVIEW": "true",
        "WOUNDCARE_BOOTSTRAP_PASSWORD": "ProfessorPreviewTestPassword!1",
        # API tests exercise the workflow, not GPU/model loading.
        "WOUNDCARE_CLS_MODEL": str(temp_db.with_suffix(".missing.pt")),
        "WOUNDCARE_SEG_MODEL": str(temp_db.with_suffix(".missing-seg.pt")),
        "WOUNDCARE_DET_MODEL": str(temp_db.with_suffix(".missing-det.pt")),
    }
)

from backend_main import app  # noqa: E402


async def _request_async(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.request(method, path, **kwargs)


def api_request(method: str, path: str, **kwargs):
    return asyncio.run(_request_async(method, path, **kwargs))


class ProfessorPreviewApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        response = api_request("POST", "/api/login", json={"username": "admin", "password": "ProfessorPreviewTestPassword!1"})
        assert response.status_code == 200, response.text
        cls.admin_headers = {"Authorization": f"Bearer {response.json()['token']}"}

    def test_preview_is_isolated_and_reports_model_identity(self) -> None:
        response = api_request("GET", "/api/health", headers=self.admin_headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["professor_preview"])
        self.assertIn("classification_model", response.json())

        patients = api_request("GET", "/api/patients", headers=self.admin_headers)
        self.assertEqual(patients.status_code, 200)
        self.assertTrue({"示範個案 A", "示範個案 B"}.issubset({patient["name"] for patient in patients.json()}))

    def test_assignment_review_gate_and_feedback(self) -> None:
        nurse_name = "preview-nurse"
        response = api_request("POST", "/api/admin/users", headers=self.admin_headers, json={"username": nurse_name, "password": "PreviewNursePassword!1", "role": "nurse"})
        self.assertEqual(response.status_code, 200, response.text)

        response = api_request(
            "POST",
            "/api/patients",
            headers=self.admin_headers,
            json={
                "room": "DEMO-03",
                "name": "示範個案 C",
                "gender": "未提供",
                "age": "50",
                "admission_date": "2026-08-18",
                "medical_history": "教授預覽資料；非真實病人。",
                "allergies": "無",
                "assigned_nurse": nurse_name,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        patient_id = response.json()["id"]

        nurse_login = api_request("POST", "/api/login", json={"username": nurse_name, "password": "PreviewNursePassword!1"})
        self.assertEqual(nurse_login.status_code, 200, nurse_login.text)
        nurse_headers = {"Authorization": f"Bearer {nurse_login.json()['token']}"}
        visible_patients = api_request("GET", "/api/patients", headers=nurse_headers)
        self.assertEqual([patient["id"] for patient in visible_patients.json()], [patient_id])

        base_record = {"shift": "白班", "treatment": "教授預覽測試紀錄", "status": "yellow", "review_status": "reviewed"}
        blocked = api_request("POST", f"/api/patients/{patient_id}/emr", headers=nurse_headers, json=base_record)
        self.assertEqual(blocked.status_code, 422)

        saved = api_request(
            "POST",
            f"/api/patients/{patient_id}/emr",
            headers=nurse_headers,
            json={**base_record, "human_review_confirmed": True},
        )
        self.assertEqual(saved.status_code, 200, saved.text)

        feedback = api_request(
            "POST",
            "/api/professor-feedback",
            headers=nurse_headers,
            json={"task": "新增傷口評估", "usability_score": 5, "clinical_clarity_score": 4, "comments": "人工覆核提示清楚。", "page": "assessment"},
        )
        self.assertEqual(feedback.status_code, 200, feedback.text)


if __name__ == "__main__":
    unittest.main()
