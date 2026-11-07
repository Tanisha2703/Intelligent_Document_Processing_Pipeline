"""
Intelligent Document Processing demo: extract fields from a receipt, with a calibrated probability per field and a human-review queue.

    streamlit run app.py

Two ways to use it:
  - "Recorded examples" works anywhere: real receipts with the fine-tuned model's stored output.
  - "Upload a receipt" runs the fine-tuned model live. It needs the Modal app deployed
    (`modal deploy experiments/serving/modal_run.py`) and `IDP_VLM_APP=idp-vlm` set; see docs/DEPLOYMENT.md.

The review tab reads and writes the same JSONL store as the HTTP service (data/review/review.jsonl).
"""

import csv
import io
import json
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples" / "recorded"
CONFIG = ROOT / "configs" / "receipt_calibration.json"
STORE_PATH = Path(os.environ.get("IDP_REVIEW_PATH", ROOT / "data" / "review" / "review.jsonl"))
GOOD, WARN = "#1baf7a", "#eb6834"

st.set_page_config(page_title="Intelligent Document Processing", page_icon="🧾", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    .field-row {display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #e8e8e4;}
    .field-name {width: 90px; color: #6b6a65; font-size: 13px; text-transform: uppercase; letter-spacing: .04em;}
    .field-value {flex: 1; font-size: 15px; word-break: break-word;}
    .chip {border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; color: white;}
    .prob {width: 52px; text-align: right; font-variant-numeric: tabular-nums; color: #3d3c39;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _store():
    from src.monitor.review import ReviewStore

    return ReviewStore(STORE_PATH)


@st.cache_resource(show_spinner="Connecting to the model...")
@st.cache_resource
def _gpu_cap():
    from src.utils.guard import cap_from_env

    return cap_from_env()


def _live_extractor():
    """The Modal-backed extractor, or None when IDP_VLM_APP is not set."""
    from src.api.backends import modal_receipts

    return modal_receipts()


def extractor():
    return st.session_state.get("extractor") or _live_extractor()


def accept_summary() -> str:
    """What the accept threshold did on the 361 SROIE test receipts, from the calibration config (never typed by hand)."""
    if os.environ.get("IDP_ACCEPT_THRESHOLD"):
        return "The accept threshold was overridden, so the measured test figures in the README do not apply."
    try:
        c = json.loads(CONFIG.read_text(encoding="utf-8"))
        return (
            f"On the 361 real SROIE test receipts this threshold accepted **{c['test_auto_accepted']:.0%}** of fields, "
            f"and **{c['test_precision_of_accepted']:.1%}** of those were correct."
        )
    except (OSError, KeyError, ValueError):
        return ""


def accept_threshold():
    try:
        if os.environ.get("IDP_ACCEPT_THRESHOLD"):
            return float(os.environ["IDP_ACCEPT_THRESHOLD"])
        return json.loads(CONFIG.read_text(encoding="utf-8"))["accept"]
    except (OSError, KeyError, ValueError):
        return None


def render_fields(result: dict, truth: dict | None = None):
    for f, v in result["fields"].items():
        auto = v["status"] == "auto"
        chip = (
            f"<span class='chip' style='background:{GOOD if auto else WARN}'>{'accepted' if auto else 'review'}</span>"
        )
        value = v["value"] if v["value"] else "<i>nothing found</i>"
        st.markdown(
            f"<div class='field-row'><div class='field-name'>{f}</div><div class='field-value'>{value}</div>"
            f"<div class='prob'>{v['probability']:.2f}</div>{chip}</div>",
            unsafe_allow_html=True,
        )
    if result["needs_review"]:
        st.warning(f"A person should check: {', '.join(result['review_fields'])}")
    else:
        st.success("Every field is above the accept threshold.")
    if truth:
        with st.expander("Dataset labels for this receipt (they contain typos)"):
            st.json(truth)


def downloads(result: dict, stem: str):
    flat = {f: v["value"] for f, v in result["fields"].items()}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value", "probability", "status"])
    for f, v in result["fields"].items():
        w.writerow([f, v["value"], f"{v['probability']:.4f}", v["status"]])
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download JSON",
        json.dumps({"fields": flat, **result}, ensure_ascii=False, indent=2),
        f"{stem}.json",
        "application/json",
    )
    c2.download_button("Download CSV", buf.getvalue(), f"{stem}.csv", "text/csv")


def extract_tab():
    mode = st.radio("Source", ["Recorded examples", "Upload a receipt"], horizontal=True, label_visibility="collapsed")
    if mode == "Recorded examples":
        examples = json.loads((EXAMPLES / "examples.json").read_text(encoding="utf-8"))
        pick = st.selectbox(
            "Receipt", range(len(examples)), format_func=lambda i: f"Receipt {i + 1}: {examples[i]['image']}"
        )
        ex = examples[pick]
        left, right = st.columns([1, 1.3])
        left.image(str(EXAMPLES / ex["image"]), use_container_width=True)
        with right:
            st.caption(
                f"Recorded output of {ex['model']}, calibrated and thresholded by the same code the service uses."
            )
            render_fields(ex["recorded_output"], ex["ground_truth"])
            downloads(ex["recorded_output"], Path(ex["image"]).stem)
        return

    up = st.file_uploader("Receipt image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    ext = extractor()
    if ext is None:
        st.info(
            "Live extraction needs the fine-tuned model deployed on Modal. Deploy it with `modal deploy experiments/serving/modal_run.py`, "
            "set `IDP_VLM_APP=idp-vlm`, and restart this app. The recorded examples work without it."
        )
        return
    password = os.environ.get("IDP_DEMO_PASSWORD")
    if password and st.text_input("Password for live extraction", type="password") != password:
        st.caption("Live extraction spends GPU time, so it is password-protected. The recorded examples are open.")
        return
    if up is None:
        return
    if st.button("Extract fields", type="primary"):
        from src.api.images import prepare_image

        if not _gpu_cap().take():
            st.error(
                "The daily limit of live extractions has been reached; it resets at 00:00 UTC. The recorded examples still work."
            )
            return
        with st.spinner("Reading the receipt (the first call after idle waits for a cold GPU, under a minute)..."):
            try:
                result = ext.extract(prepare_image(up.getvalue()))
            except Exception as exc:
                st.error(f"Could not process this receipt: {type(exc).__name__}: {exc}")
                return
        if result["needs_review"]:
            result["doc_id"] = _store().enqueue(
                None, result["doc_type"], {f: {**v, "source": "vlm"} for f, v in result["fields"].items()}
            )
        st.session_state["last"] = (up.name, result, up.getvalue())
    last = st.session_state.get("last")
    if last:
        name, result, data = last
        left, right = st.columns([1, 1.3])
        left.image(data, use_container_width=True)
        with right:
            render_fields(result)
            downloads(result, Path(name).stem)


def review_tab():
    store = _store()
    stats = store.stats()
    a, b, c = st.columns(3)
    a.metric("Documents waiting", stats["pending_docs"])
    b.metric("Fields resolved", stats["resolved_fields"])
    c.metric(
        "Model was right",
        f"{stats['model_was_right']} of {stats['resolved_fields']}" if stats["resolved_fields"] else "n/a",
    )
    pending = store.pending()
    if not pending:
        st.info("Nothing to review. Fields the model is unsure about, from live extractions, appear here.")
        return
    for doc in pending:
        st.markdown(f"**Document {doc['doc_id']}**")
        for f, v in doc["fields"].items():
            col1, col2, _ = st.columns([3, 1, 1])
            new = col1.text_input(
                f"{f} (model confidence {v['probability']:.2f})", value=v["value"] or "", key=f"v-{doc['doc_id']}-{f}"
            )
            if col2.button("Save", key=f"s-{doc['doc_id']}-{f}"):
                store.resolve(doc["doc_id"], f, new or None, "streamlit")
                st.rerun()


def main():
    st.markdown("## Intelligent Document Processing")
    st.caption(
        "Receipt fields with a calibrated probability each. A fine-tuned Qwen2.5-VL reads the receipt; "
        "fields it is unsure about go to a person."
    )
    with st.sidebar:
        st.markdown("### How to read this")
        thr = accept_threshold()
        st.markdown(
            "Each field has a probability that it is correct. Fields at or above the accept threshold"
            + (f" (**{thr:.2f}**)" if thr else "")
            + " are accepted automatically; the rest wait for review."
        )
        st.markdown(
            accept_summary() + " Addresses are the weakest field. See the README for the full results and limits."
        )
    tab1, tab2 = st.tabs(["Extract", "Review queue"])
    with tab1:
        extract_tab()
    with tab2:
        review_tab()


main()
