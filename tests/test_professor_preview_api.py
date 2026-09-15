from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys
import asyncio
from unittest.mock import patch

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
import backend_main as backend  # noqa: E402


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
        for username, role in (("review-nurse", "nurse"), ("review-head", "head_nurse")):
            created = api_request("POST", "/api/admin/users", headers=cls.admin_headers, json={"username": username, "password": "SyntheticReviewPassword!1", "role": role})
            assert created.status_code == 200, created.text
            login = api_request("POST", "/api/login", json={"username": username, "password": "SyntheticReviewPassword!1"})
            setattr(cls, "nurse_headers" if role == "nurse" else "head_headers", {"Authorization": f"Bearer {login.json()['token']}"})

    def test_preview_is_isolated_and_reports_model_identity(self) -> None:
        response = api_request("GET", "/api/health", headers=self.admin_headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["professor_preview"])
        self.assertIn("classification_model", response.json())
        self.assertFalse(response.json()["inference_ready"])
        self.assertFalse(response.json()["model_weights_learn_from_feedback"])

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

    def guidance(self, **overrides):
        return {"title": "合成測試建議", "wound_classes": "Abrasions", "recommendation": "這是測試用文字，不是臨床處置指示。", "source": "synthetic test protocol", "review_confirmed": True, "deidentified_confirmed": True, **overrides}

    def test_rag_confirmation_and_role_required(self):
        for field in ("review_confirmed", "deidentified_confirmed"):
            response = api_request("POST", "/api/rag/guidance", headers=self.head_headers, json=self.guidance(**{field: False}))
            self.assertEqual(response.status_code, 422)
        response = api_request("POST", "/api/rag/guidance", headers=self.nurse_headers, json=self.guidance())
        self.assertEqual(response.status_code, 403)
        pending = api_request("GET", "/api/rag/guidance?include_pending=true", headers=self.nurse_headers)
        self.assertEqual(pending.status_code, 403)

    def test_rag_missing_source_and_whitespace_rejected(self):
        for change, expected in (({"source_emr_id": 9999999}, 404), ({"title": "   "}, 422), ({"recommendation": " " * 12}, 422), ({"source": "  "}, 422)):
            response = api_request("POST", "/api/rag/guidance", headers=self.head_headers, json=self.guidance(**change))
            self.assertEqual(response.status_code, expected, response.text)

    def test_rag_reviewed_source_persisted_and_emr_id_correct(self):
        patient = api_request("GET", "/api/patients", headers=self.admin_headers).json()[0]
        # Make reports and EMR auto-increment sequences deliberately different.
        with backend.db() as connection:
            for _ in range(3):
                connection.execute("INSERT INTO reports(nurse_name, patient_name, action) VALUES ('synthetic', 'synthetic', 'test')")
        record = {"shift": "白班", "treatment": "合成測試紀錄", "status": "yellow", "review_status": "needs_review"}
        saved = api_request("POST", f"/api/patients/{patient['id']}/emr", headers=self.head_headers, json=record)
        self.assertEqual(saved.status_code, 200, saved.text)
        emr_id = saved.json()["id"]
        with backend.db() as connection:
            self.assertIsNotNone(connection.execute("SELECT id FROM emr_records WHERE id = ?", (emr_id,)).fetchone())
        blocked = api_request("POST", "/api/rag/guidance", headers=self.head_headers, json=self.guidance(source_emr_id=emr_id))
        self.assertEqual(blocked.status_code, 422)
        saved = api_request("POST", f"/api/patients/{patient['id']}/emr", headers=self.head_headers, json={**record, "review_status": "reviewed", "human_review_confirmed": True})
        accepted = api_request("POST", "/api/rag/guidance", headers=self.head_headers, json=self.guidance(source_emr_id=saved.json()["id"]))
        self.assertEqual(accepted.status_code, 200, accepted.text)
        with backend.db() as connection:
            row = connection.execute("SELECT source_emr_id, reviewed_by, reviewed_at FROM rag_guidance WHERE id = ?", (accepted.json()["id"],)).fetchone()
            self.assertEqual(row["source_emr_id"], saved.json()["id"])
            self.assertIsNotNone(row["reviewed_by"])
            self.assertIsNotNone(row["reviewed_at"])

    def test_rag_legacy_hidden_until_explicit_review_and_revocation(self):
        with backend.db() as connection:
            admin_id = connection.execute("SELECT id FROM users WHERE username='admin'").fetchone()[0]
            legacy_id = connection.execute("INSERT INTO rag_guidance(title, wound_classes, recommendation, created_by) VALUES ('舊版合成建議', 'Abrasions', '這是舊版合成測試文字。', ?)", (admin_id,)).lastrowid
        def visible_ids():
            return {row["id"] for row in api_request("GET", "/api/rag/guidance?limit=20", headers=self.nurse_headers).json()}
        self.assertNotIn(legacy_id, visible_ids())
        pending = api_request("GET", "/api/rag/guidance?include_pending=true&limit=20", headers=self.head_headers).json()
        self.assertIn(legacy_id, {row["id"] for row in pending})
        endpoint = f"/api/rag/guidance/{legacy_id}"
        self.assertEqual(api_request("PATCH", endpoint, headers=self.head_headers, json={"is_active": True}).status_code, 422)
        self.assertEqual(api_request("PATCH", endpoint, headers=self.head_headers, json={"is_active": True, "review_confirmed": True, "deidentified_confirmed": True}).status_code, 200)
        self.assertIn(legacy_id, visible_ids())
        self.assertEqual(api_request("PATCH", endpoint, headers=self.head_headers, json={"is_active": False}).status_code, 200)
        self.assertNotIn(legacy_id, visible_ids())
        with backend.db() as connection:
            self.assertIsNotNone(connection.execute("SELECT id FROM rag_guidance WHERE id = ?", (legacy_id,)).fetchone())

    def test_patient_access_and_empty_upload_fail_closed(self):
        patient = api_request("GET", "/api/patients", headers=self.admin_headers).json()[0]
        self.assertEqual(api_request("GET", f"/api/patients/{patient['id']}/emr", headers=self.nurse_headers).status_code, 403)
        self.assertEqual(api_request("POST", "/api/predict", headers=self.admin_headers, files={"file": ("empty.png", b"", "image/png")}).status_code, 400)
        self.assertEqual(api_request("GET", "/api/health").status_code, 401)

    def test_high_confidence_without_roi_never_retrieves_care(self):
        result = self.synthetic_prediction(roi=False, confidence=.9932)
        self.assertTrue(result["advice_blocked"])
        self.assertFalse(result["localization_available"])
        self.assertEqual(result["rag_guidance"], [])
        self.assertEqual(result["classes"], ["Bruises"])
        self.assertIn("no_reliable_roi", result["review_reasons"])
        self.assertIn("高分類信心不代表", result["llm_advice"])

    def test_uncertain_localized_prediction_never_retrieves_care(self):
        result = self.synthetic_prediction(roi=True, confidence=.3)
        self.assertTrue(result["advice_blocked"])
        self.assertEqual(result["rag_guidance"], [])
        self.assertTrue(result["requires_human_review"])

    def test_reliable_localized_prediction_still_requires_review(self):
        result = self.synthetic_prediction(roi=True, confidence=.9)
        self.assertFalse(result["advice_blocked"])
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(len(result["rag_guidance"]), 1)

    def synthetic_prediction(self, *, roi, confidence):
        image = backend.np.zeros((32, 32, 3), dtype=backend.np.uint8)
        _, png = backend.cv2.imencode('.png', image)
        value = {"mode": "full_image_primary", "fallback_used": False, "segment_confidence": .8 if roi else None,
                 "class_confidence": confidence, "crop_box": {"x1": 2, "y1": 2, "x2": 20, "y2": 20} if roi else None,
                 "low_confidence": confidence < .6, "label": "Bruises"}
        with patch.object(backend, "seg_model", object()), patch.object(backend, "cls_model", object()), \
                patch.object(backend, "infer_segmentation_cascade", return_value=value), \
                patch.object(backend, "retrieve_rag_guidance", return_value=[{"id": 1, "title": "synthetic"}]) as retrieval:
            response = api_request("POST", "/api/predict", headers=self.admin_headers, files={"file": ("synthetic.png", png.tobytes(), "image/png")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(retrieval.call_count, int(roi and confidence >= .6))
            return response.json()


if __name__ == "__main__":
    unittest.main()
