"""
Multilingual Document Intelligence — Streamlit Application

Drag-and-drop interface for Hindi/English document processing.
  Upload PDF/PNG/JPG/TIFF → OCR → NER → Classification → Export

Entry point: streamlit run app.py
"""
import io
import logging
import time
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Stat box */
    .stat-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #212529;
    }

    /* Entity tag */
    .entity-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        white-space: nowrap;
    }

    /* Confidence bar */
    .conf-bar-outer {
        background: #e9ecef;
        border-radius: 6px;
        height: 8px;
        margin-top: 4px;
    }
    .conf-bar-inner {
        border-radius: 6px;
        height: 8px;
    }

    /* Card */
    .card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Hero header */
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #6c757d;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Pipeline loader (cached — loads models once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipeline():
    """Load and return the DocumentPipeline singleton."""
    from src.pipeline.document_pipeline import DocumentPipeline
    pipeline = DocumentPipeline()
    pipeline.initialize()
    return pipeline


# ---------------------------------------------------------------------------
# Helper: render entity section
# ---------------------------------------------------------------------------
def _conf_color(conf: float) -> str:
    if conf >= 0.80:
        return "#28a745"
    elif conf >= 0.50:
        return "#ffc107"
    return "#dc3545"


def render_entities(entities_by_type: dict, show_confidence: bool, threshold: float):
    """Render entities grouped by type with colored tags."""
    from src.ner.entities import ENTITY_COLORS, ENTITY_DISPLAY_NAMES, ENTITY_ICONS

    has_any = False
    for etype, ents in entities_by_type.items():
        ents = [e for e in ents if e.get("confidence", 0) >= threshold]
        if not ents:
            continue
        has_any = True
        color = ENTITY_COLORS.get(etype, "#e9ecef")
        icon = ENTITY_ICONS.get(etype, "•")
        display_name = ENTITY_DISPLAY_NAMES.get(etype, etype)

        st.markdown(f"**{icon} {display_name}** ({len(ents)})")
        tag_html = ""
        for ent in ents:
            conf_pct = f" <span style='font-size:11px;color:#666'>({ent['confidence']:.0%})</span>" if show_confidence else ""
            tag_html += (
                f"<span class='entity-tag' style='background:{color}'>"
                f"{ent['value']}{conf_pct}"
                f"</span>"
            )
        st.markdown(tag_html, unsafe_allow_html=True)

    if not has_any:
        st.info("No entities found above confidence threshold.")


# ---------------------------------------------------------------------------
# Helper: render confidence bar
# ---------------------------------------------------------------------------
def confidence_bar(value: float, label: str = "") -> str:
    color = _conf_color(value)
    pct = int(value * 100)
    return (
        f"<div>{label} <strong>{pct}%</strong></div>"
        f"<div class='conf-bar-outer'>"
        f"<div class='conf-bar-inner' style='width:{pct}%;background:{color}'></div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Header ----------------------------------------------------------
    st.markdown(
        "<div class='hero-title'>📄 Multilingual Document Intelligence</div>"
        "<div class='hero-sub'>Extract structured data from Hindi and English documents "
        "using fine-tuned transformer models</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- Sidebar ---------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        show_raw = st.checkbox("Show raw OCR text", value=True)
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_all_cls_scores = st.checkbox("Show all classification scores", value=False)
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Entities below this confidence are hidden",
        )
        st.divider()
        st.markdown("### 📊 About")
        st.markdown(
            "**Pipeline**: OCR → NER → Classification\n\n"
            "**Models**: XLM-RoBERTa (fine-tuned)\n\n"
            "**Languages**: Hindi + English + Mixed\n\n"
            "**Accuracy**: 92%+ on clean 300 DPI scans\n\n"
            "**Export**: JSON / CSV / Excel"
        )

    # ---- File uploader ---------------------------------------------------
    uploaded_file = st.file_uploader(
        "📤 Drag and drop your document here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        help="Supported formats: PDF, PNG, JPG, TIFF — Hindi, English, or mixed language",
    )

    process_btn = st.button("🚀 Process Document", type="primary", disabled=uploaded_file is None)

    # ---- Processing ------------------------------------------------------
    if process_btn and uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_type = Path(uploaded_file.name).suffix.lstrip(".").lower()

        with st.spinner("Processing document — this may take 10–30 seconds..."):
            try:
                pipeline = load_pipeline()
                result = pipeline.process(
                    file_bytes=file_bytes,
                    file_type=file_type,
                    confidence_threshold=threshold,
                )
                st.session_state["result"] = result
                st.session_state["file_name"] = uploaded_file.name
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                logger.exception("Pipeline error")
                return

    # ---- Results ---------------------------------------------------------
    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            "<br><br><div style='text-align:center;color:#adb5bd;font-size:1.1rem'>"
            "Upload a document and click <b>Process Document</b> to begin."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ---- Stats row -------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='stat-box'><div class='stat-label'>Processing Time</div>"
            f"<div class='stat-value'>{result['processing_time_s']}s</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-box'><div class='stat-label'>OCR Confidence</div>"
            f"<div class='stat-value'>{result['ocr_confidence']:.0f}%</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        lang = result["language"].get("primary", "unknown").capitalize()
        st.markdown(
            f"<div class='stat-box'><div class='stat-label'>Language</div>"
            f"<div class='stat-value'>{lang}</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='stat-box'><div class='stat-label'>Entities Found</div>"
            f"<div class='stat-value'>{result['entity_count']}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Classification + Entities (2-column) ---------------------------
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("#### 📋 Classification")
        classification = result.get("classification", {})
        display = classification.get("display", "❓ Unknown")
        conf = classification.get("confidence", 0.0)
        st.markdown(f"**Category:** {display}")
        st.markdown(
            confidence_bar(conf, "Confidence"),
            unsafe_allow_html=True,
        )

        if show_all_cls_scores:
            st.markdown("**All category scores:**")
            all_scores = classification.get("all_scores", {})
            for cat, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                st.markdown(
                    f"<small>{cat}</small> {confidence_bar(score)}",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 🌐 Language")
        language = result.get("language", {})
        hi = language.get("hindi_ratio", 0.0)
        en = language.get("english_ratio", 0.0)
        st.markdown(
            f"Hindi: **{hi:.0%}** &nbsp; English: **{en:.0%}**",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<small>Pages: {result.get('page_count', 1)} | "
            f"Threshold: {threshold:.0%}</small>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("#### 🏷️ Extracted Entities")
        render_entities(
            result.get("entities_by_type", {}),
            show_confidence=show_confidence,
            threshold=threshold,
        )

    # ---- Raw OCR Text ----------------------------------------------------
    if show_raw:
        st.markdown("---")
        st.markdown("#### 📝 Raw OCR Text")
        ocr_text = result.get("ocr_text", "")
        if ocr_text.strip():
            st.text_area(
                label="OCR output",
                value=ocr_text,
                height=250,
                label_visibility="collapsed",
            )
        else:
            st.warning("No text extracted — check document quality or Tesseract installation.")

    # ---- Export ----------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💾 Export Results")

    file_stem = Path(st.session_state.get("file_name", "document")).stem

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        from src.utils.export import to_json
        json_str = to_json(result)
        st.download_button(
            label="⬇️ JSON",
            data=json_str.encode("utf-8"),
            file_name=f"{file_stem}_result.json",
            mime="application/json",
        )

    with exp_col2:
        from src.utils.export import to_csv
        csv_str = to_csv(result)
        st.download_button(
            label="⬇️ CSV",
            data=csv_str.encode("utf-8"),
            file_name=f"{file_stem}_entities.csv",
            mime="text/csv",
        )

    with exp_col3:
        from src.utils.export import to_excel_bytes
        excel_bytes = to_excel_bytes(result)
        st.download_button(
            label="⬇️ Excel",
            data=excel_bytes,
            file_name=f"{file_stem}_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
