import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import SessionLocal
from app.database.models.prediction import Prediction
from app.database.models.feedback import Feedback
from app.database.models.user import User
from app.database.models.model_version import ModelVersion

def test_admin_flow():
    print("=== Starting E2E Verification of Fake News Admin Panel ===")
    client = TestClient(app)

    with SessionLocal() as db:
        # Ensure a dummy user and model version exist for testing
        user = db.query(User).filter(User.username == "admin_test_user").first()
        if not user:
            user = User(username="admin_test_user", email="admin_test@example.com", password_hash="hashed_pw")
            db.add(user)
            db.commit()
            db.refresh(user)

        model_ver = db.query(ModelVersion).first()
        if not model_ver:
            model_ver = ModelVersion(
                version_str="1.0.1",
                model_name="svm",
                algorithm_name="Linear SVM",
                filepath="ml/training/models/svm/model.joblib",
                is_active=True
            )
            db.add(model_ver)
            db.commit()
            db.refresh(model_ver)

        # Create a test prediction
        pred = Prediction(
            title="Breaking: Mars Colony Discovered",
            text_content="Scientists have discovered a secret colony on Mars built in 1990.",
            predicted_label="FAKE",
            confidence_score=0.94,
            explanation="Unverified claims found.",
            model_version_id=model_ver.id,
            user_id=user.id
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)
        print(f"[1] Created test prediction #{pred.id}")

        # Create a test pending feedback
        feedback = Feedback(
            prediction_id=pred.id,
            user_id=user.id,
            is_correct=False, # User disagrees (believes it's REAL)
            user_comment="This news article is actually genuine research.",
            status="PENDING"
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        fb_id = feedback.id
        print(f"[2] Created test pending feedback #{fb_id} with status='PENDING'")

    # 1. Test GET /api/v1/admin/pending
    response = client.get("/api/v1/admin/pending")
    assert response.status_code == 200, f"GET pending failed: {response.text}"
    pending_list = response.json().get("pending_feedbacks", [])
    found_item = next((item for item in pending_list if item.get("Feedback ID") == str(fb_id)), None)
    assert found_item is not None, f"Feedback #{fb_id} not found in pending list!"
    print(f"[3] GET /api/v1/admin/pending verified: Feedback #{fb_id} successfully listed.")

    # 2. Test POST /api/v1/admin/review (ACCEPT / APPROVED)
    review_payload = {
        "feedback_id": str(fb_id),
        "review_status": "ACCEPT", # Testing normalization of ACCEPT -> APPROVED
        "reviewer": "Test Admin",
        "notes": "Verified against external fact-check source."
    }
    rev_resp = client.post("/api/v1/admin/review", json=review_payload)
    assert rev_resp.status_code == 200, f"POST review failed: {rev_resp.text}"
    print(f"[4] POST /api/v1/admin/review verified: Review status updated.")

    # Verify status updated in database to APPROVED
    with SessionLocal() as db:
        updated_fb = db.query(Feedback).filter(Feedback.id == fb_id).first()
        assert updated_fb.status == "APPROVED", f"DB status is {updated_fb.status}, expected APPROVED!"
        print(f"[5] PostgreSQL Database verified: Feedback #{fb_id} status is now 'APPROVED'.")

    # Verify feedback is no longer in pending list
    response2 = client.get("/api/v1/admin/pending")
    pending_list2 = response2.json().get("pending_feedbacks", [])
    found_item2 = next((item for item in pending_list2 if item.get("Feedback ID") == str(fb_id)), None)
    assert found_item2 is None, f"Feedback #{fb_id} still in pending list after approval!"
    print(f"[6] Pending queue verified: Approved feedback automatically cleared from pending list.")

    # 3. Test REJECT status
    with SessionLocal() as db:
        feedback2 = Feedback(
            prediction_id=pred.id,
            user_id=user.id,
            is_correct=True,
            user_comment="Spam comment to test reject flow.",
            status="PENDING"
        )
        db.add(feedback2)
        db.commit()
        db.refresh(feedback2)
        fb_id2 = feedback2.id

    reject_payload = {
        "feedback_id": str(fb_id2),
        "review_status": "REJECTED",
        "reviewer": "Test Admin",
        "notes": "Spam submission rejected."
    }
    rej_resp = client.post("/api/v1/admin/review", json=reject_payload)
    assert rej_resp.status_code == 200, f"POST review reject failed: {rej_resp.text}"

    with SessionLocal() as db:
        rej_fb = db.query(Feedback).filter(Feedback.id == fb_id2).first()
        assert rej_fb.status == "REJECTED", f"DB status is {rej_fb.status}, expected REJECTED!"
        print(f"[7] Rejection flow verified: Feedback #{fb_id2} status is now 'REJECTED'.")

    print("=========================================================")
    print("SUCCESS: ALL ADMIN PANEL & DATABASE FLOWS VERIFIED 100%")
    print("=========================================================")

if __name__ == "__main__":
    test_admin_flow()
