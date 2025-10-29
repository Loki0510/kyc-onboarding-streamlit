"""
Enhanced Onboarding KYC - Streamlit Web App (single-file)

Features:
- Multi-step onboarding for New or Existing customers
- Document upload with optional OCR (pytesseract)
- Liveness check via webcam (st.camera_input)
- Signature collection via streamlit-drawable-canvas
- Mock AI automation hooks for OCR/face-match/risk scoring (placeholders)
- Simple security-minded UI hints (PII masking, do-not-store notes)

To run:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
import io
import base64
import time
import json
import numpy as np
import cv2
from datetime import datetime

# Optional imports - gracefully degrade if not installed
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# face-recognition optional (heavy). If not available, we'll mock.
try:
    import face_recognition
    FACE_AVAILABLE = True
except Exception:
    FACE_AVAILABLE = False

# ---------- App configuration ----------
st.set_page_config(page_title="Enhanced Onboarding KYC", layout="wide")
st.markdown("<style> .stApp { background-color: #f7f9fc; } </style>", unsafe_allow_html=True)

# ---------- Helper functions ----------
def mask_ssn(ssn: str) -> str:
    """Simple PII mask for SSN-like strings"""
    if not ssn: return ""
    return "***-**-" + ssn[-4:] if len(ssn) >= 4 else "***"

def image_bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_image_to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# Mock AI functions - replace with real services
def mock_ocr_extract_text(image: Image.Image):
    """Try local pytesseract if available, otherwise return placeholder."""
    if TESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"[OCR error: {e}]"
    else:
        return "[OCR not installed - configure pytesseract or plug cloud OCR]"

def mock_face_match(photo_image: Image.Image, selfie_image: Image.Image):
    """Return a mocked confidence score and boolean match. Replace with real face match API."""
    if FACE_AVAILABLE:
        try:
            # Convert PIL -> numpy array
            a = np.array(photo_image)
            b = np.array(selfie_image)
            enc1 = face_recognition.face_encodings(a)
            enc2 = face_recognition.face_encodings(b)
            if not enc1 or not enc2:
                return {"match": False, "confidence": 0.0, "reason": "face_not_found"}
            dist = np.linalg.norm(enc1[0] - enc2[0])
            # small distance = high similarity; convert heuristically to confidence
            confidence = max(0.0, 1.0 - dist/0.6)  # arbitrary mapping
            return {"match": confidence > 0.45, "confidence": round(float(confidence), 2)}
        except Exception as e:
            return {"match": False, "confidence": 0.0, "reason": f"face_error:{e}"}
    else:
        # Simple mock: always high confidence for demo
        return {"match": True, "confidence": 0.92, "note": "mocked - install face-recognition or use cloud API"}

def mock_risk_assessment(metadata: dict):
    """Return a simple risk score and suggestions. Replace with real risk engine."""
    score = 0.1
    reasons = []
    if metadata.get("is_existing") is False:
        score += 0.15
    # example: older documents -> raise score (mock)
    if metadata.get("document_type", "") == "passport":
        score -= 0.02
    return {"risk_score": round(min(max(score, 0.0), 1.0), 2), "reasons": reasons}

def save_user_bundle(bundle: dict) -> str:
    """
    Placeholder for saving data securely to storage.
    DO NOT store unencrypted PII in plaintext.
    Replace this function with secure server-side storage + encryption key management.
    """
    # For demo only: return a fake storage id
    return "mock-storage-id-12345"

# ---------- Session state initialization ----------
if "step" not in st.session_state:
    st.session_state.step = 1
if "flow_for" not in st.session_state:
    st.session_state.flow_for = "new"  # 'new' or 'existing'
if "collected" not in st.session_state:
    st.session_state.collected = {}

# ---------- Top-level layout ----------
st.header("Enhanced Onboarding KYC — Bank of America (Prototype)")
st.write("A demo of a modern KYC flow emphasizing AI automation, UX, security, and document verification innovations.")
st.write("---")

# Left sidebar for quick navigation
with st.sidebar:
    st.subheader("Start")
    st.radio("Customer type", ("New customer", "Existing customer"), index=0 if st.session_state.flow_for == "new" else 1,
             key="flow_type_radio", on_change=lambda: st.session_state.update({"flow_for": "new" if st.session_state.flow_type_radio == "New customer" else "existing"}))
    st.markdown("### Progress")
    steps_total = 4
    st.progress((st.session_state.step - 1) / steps_total)
    st.markdown("---")
    st.markdown("Help & Security")
    st.info("No files are stored unencrypted in this demo. In production, use server-side encryption + secure key management.")
    st.markdown("**Tip:** Use a government ID (driver's license or passport) for faster verification.")

# ---------- Stepper UI (Main area) ----------

def show_step_1_personal_info():
    st.subheader("1) Personal Information")
    st.write("Enter minimal PII. We mask sensitive details client-side for display.")
    cols = st.columns([2, 1])
    
    # Helper to parse DOB string to date object
    def parse_dob(dob_str):
        if not dob_str:
            return None
        try:
            return datetime.strptime(dob_str, "%Y-%m-%d").date()
        except:
            return None

    with cols[0]:
        first = st.text_input("First name", value=st.session_state.collected.get("first_name", ""))
        last = st.text_input("Last name", value=st.session_state.collected.get("last_name", ""))
        
        # Fix: Parse stored string back to date
        current_dob_str = st.session_state.collected.get("dob", "")
        dob_value = parse_dob(current_dob_str)
        dob = st.date_input("Date of birth", value=dob_value)

        ssn = st.text_input("SSN (last 4 digits only recommended)", value=st.session_state.collected.get("ssn", ""))

    with cols[1]:
        phone = st.text_input("Phone", value=st.session_state.collected.get("phone", ""))
        email = st.text_input("Email", value=st.session_state.collected.get("email", ""))
        st.write("Customer type:")
        st.radio("type", ("Individual", "Business"), index=0, key="cust_type_radio")
    
    # Save into state: store as string for consistency
    st.session_state.collected.update({
        "first_name": first,
        "last_name": last,
        "dob": dob.strftime("%Y-%m-%d") if dob else "",  # Save as string
        "ssn": ssn,
        "phone": phone,
        "email": email,
        "is_existing": (st.session_state.flow_for == "existing")
    })

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.step > 1:
            if st.button("Back", key="back_1"):
                st.session_state.step = max(1, st.session_state.step - 1)
    with c2:
        if st.button("Continue", type="primary", key="continue_1"):
            st.session_state.step = 2

def show_step_2_document_upload():
    st.subheader("2) Identity Document Upload & AI OCR")
    st.write("Upload a government-issued ID (driver's license or passport). We will attempt OCR to extract name and DOB automatically.")
    st.info("Allowed: PNG/JPEG/PDF. For PDF, only first page will be processed (if local OCR is used).")

    doc_type = st.selectbox("Document type", ["Driver's License", "Passport", "State ID"], index=0)
    uploaded = st.file_uploader("Upload front of ID", type=["png","jpg","jpeg","pdf"], key="id_front")
    uploaded_back = st.file_uploader("Upload back of ID (optional)", type=["png","jpg","jpeg","pdf"], key="id_back")

    extracted_text = None
    if uploaded:
        raw = uploaded.read()
        try:
            # If PDF, read first page image via PIL fallback (not robust) - for demo we treat as image
            front_img = image_bytes_to_pil(raw)
            st.image(front_img, caption="ID front (preview)", use_column_width=False)
            st.session_state.collected["id_front_image"] = pil_image_to_bytes(front_img)
            st.session_state.collected["document_type"] = doc_type.lower().replace(" ", "_")
            # OCR
            with st.spinner("Running OCR (local or mocked)..."):
                extracted_text = mock_ocr_extract_text(front_img)
                st.session_state.collected["ocr_text"] = extracted_text
            st.markdown("**OCR extract (preview):**")
            st.code(extracted_text[:800] + ("..." if extracted_text and len(extracted_text) > 800 else ""))
        except Exception as e:
            st.error(f"Could not preview file: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="back_2"):
            st.session_state.step = 1
    with c2:
        if st.button("Continue", key="continue_2"):
            # quick validation
            if not uploaded:
                st.warning("Please upload an ID to continue.")
            else:
                st.session_state.step = 3

def show_step_3_liveness_and_face_match():
    st.subheader("3) Liveness Check & Face Match")
    st.write("Use your webcam to capture a live selfie. We'll run liveness heuristics and match it to the ID photo.")
    st.write("Liveness uses short movement prompts, optional liveness challenges or blink detection in production.")

    selfie = st.camera_input("Take a selfie (uses your webcam)")
    if selfie:
        selfie_bytes = selfie.getvalue()
        selfie_img = image_bytes_to_pil(selfie_bytes)
        st.image(selfie_img, caption="Selfie (preview)", width=240)
        st.session_state.collected["selfie_image"] = pil_image_to_bytes(selfie_img)

    st.markdown("**Optional:** Upload a higher-quality selfie.")
    selfie_upload = st.file_uploader("Or upload selfie", type=["png","jpg","jpeg"], key="selfie_upload")
    if selfie_upload and not selfie:
        selfie_bytes = selfie_upload.read()
        selfie_img = image_bytes_to_pil(selfie_bytes)
        st.image(selfie_img, caption="Uploaded selfie (preview)", width=240)
        st.session_state.collected["selfie_image"] = pil_image_to_bytes(selfie_img)

    st.divider()
    if st.button("Run face-match (AI)"):
        if "id_front_image" not in st.session_state.collected or "selfie_image" not in st.session_state.collected:
            st.warning("Need both ID front and selfie to run face match.")
        else:
            # convert bytes -> PIL images
            id_img = image_bytes_to_pil(st.session_state.collected["id_front_image"])
            selfie_img = image_bytes_to_pil(st.session_state.collected["selfie_image"])
            with st.spinner("Comparing faces..."):
                res = mock_face_match(id_img, selfie_img)
                st.session_state.collected["face_match"] = res
                st.success(f"Face match result: {res}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="back_3"):
            st.session_state.step = 2
    with c2:
        if st.button("Continue", key="continue_3"):
            # basic validation
            if "face_match" not in st.session_state.collected:
                st.warning("Please run face match before continuing.")
            else:
                st.session_state.step = 4

def show_step_4_signature_and_submit():
    st.subheader("4) Digital Signature & Consent")
    st.write("Sign below to electronically agree to terms (e-signature).")
    if CANVAS_AVAILABLE:
        canvas_result = st_canvas(
            stroke_width=2,
            stroke_color="#000",
            background_color="#fff",
            height=200,
            width=700,
            drawing_mode="freedraw",
            key="sig_canvas"
        )
        if canvas_result and canvas_result.image_data is not None:
            # store PNG bytes
            img_arr = (canvas_result.image_data * 255).astype(np.uint8)
            pil_sig = Image.fromarray(img_arr)
            sig_bytes = pil_image_to_bytes(pil_sig)
            st.session_state.collected["signature"] = sig_bytes
            st.image(pil_sig, caption="Signature preview", width=300)
    else:
        st.warning("Signature canvas component not installed. Install `streamlit-drawable-canvas` to capture signature.")
        st.markdown("Or upload an image of your signature:")
        uploaded_sig = st.file_uploader("Upload signature image", type=["png","jpg","jpeg"], key="sig_upload")
        if uploaded_sig:
            sig_img = image_bytes_to_pil(uploaded_sig.read())
            st.session_state.collected["signature"] = pil_image_to_bytes(sig_img)
            st.image(sig_img, caption="Signature preview", width=300)

    st.checkbox("I consent to electronic signature and verification", key="consent_check")

    st.write("Final review (masked PII shown):")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Full name:")
        st.info(f"{st.session_state.collected.get('first_name','')} {st.session_state.collected.get('last_name','')}")
        st.write("Email / Phone:")
        st.info(f"{st.session_state.collected.get('email','')} / {st.session_state.collected.get('phone','')}")
    with col2:
        st.write("SSN (masked):")
        st.info(mask_ssn(st.session_state.collected.get("ssn","")))
        st.write("Document type:")
        st.info(st.session_state.collected.get("document_type",""))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="back_4"):
            st.session_state.step = 3
    with c2:
        if st.button("Submit & Verify", key="submit_4"):
            if not st.session_state.get("consent_check", False):
                st.warning("Consent required to proceed.")
            else:
                # run risk assessment + final AI checks
                with st.spinner("Running final checks..."):
                    meta = {
                        "is_existing": st.session_state.collected.get("is_existing", False),
                        "document_type": st.session_state.collected.get("document_type","")
                    }
                    risk = mock_risk_assessment(meta)
                    st.session_state.collected["risk"] = risk
                    # mock save
                    storage_id = save_user_bundle(st.session_state.collected)
                    st.session_state.collected["storage_id"] = storage_id
                    time.sleep(1.2)
                st.success("KYC submitted successfully.")
                st.balloons()
                st.markdown("### Summary")
                st.write("**Risk score:**", risk["risk_score"])
                st.write("**Face match:**", st.session_state.collected.get("face_match", {}))
                st.write("**OCR excerpt:**")
                ocr = st.session_state.collected.get("ocr_text", "[none]")
                st.code(ocr[:1000] + ("..." if len(ocr) > 1000 else ""))
                st.write("**Storage reference (demo):**", storage_id)
                st.info("In production, follow strict retention & encryption policies; do not keep PII longer than necessary.")
                # Advance to 'done'
                st.session_state.step = 5

# ---------- Main flow controller ----------
if st.session_state.step == 1:
    show_step_1_personal_info()
elif st.session_state.step == 2:
    show_step_2_document_upload()
elif st.session_state.step == 3:
    show_step_3_liveness_and_face_match()
elif st.session_state.step == 4:
    show_step_4_signature_and_submit()
elif st.session_state.step == 5:
    st.header("KYC Completed — Next Steps")
    st.write("Thank you. The submitted KYC will be reviewed automatically and by our compliance team if flagged.")
    st.write("If additional documentation is required, we will notify you via email.")
    if st.button("Start new flow"):
        st.session_state.step = 1
        st.session_state.collected = {}
    if st.button("Download submission (JSON)"):
        # Create a safe JSON with masked PII for download
        masked = dict(st.session_state.collected)
        masked["ssn"] = mask_ssn(masked.get("ssn",""))
        masked["email"] = masked.get("email","")
        # remove raw image bytes from download for security (demo)
        for k in ["id_front_image","selfie_image","signature"]:
            if k in masked: masked[k] = f"<binary removed ({k})>"
        payload = json.dumps(masked, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="kyc_submission.json">Click to download JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

# ---------- Footer / small notes ----------
st.write("---")
st.caption("Prototype — not for production. Replace mocks with secure OCR, liveness detection, and identity resolution services. Ensure compliance with privacy & AML regulations before deploying.")
