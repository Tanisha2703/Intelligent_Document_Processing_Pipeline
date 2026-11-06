"""
Routing: from a document type to the extractor that can handle it.

Only types an extractor was fine-tuned and measured on are routed. Everything else, and every document whose
calibrated type confidence is below the accept threshold, goes to a person. An unmeasured guess dressed as a
route would be worse than saying "not supported".
"""

# document type -> extractor. receipts: SROIE/CORD-style fields; forms: XFUND key-value pairs (a weaker model)
ROUTES = {"receipt": "receipt", "form": "form", "questionnaire": "form"}


def route(prediction: dict, accept: float) -> dict:
    """prediction: Probe.predict() output. Returns the prediction plus `route` and `reason`."""
    doc_type = prediction["doc_type"]
    if prediction["confidence"] < accept:
        out = {
            "route": "human",
            "reason": f"unsure of the document type (confidence {prediction['confidence']:.2f} below {accept:.2f})",
        }
    elif doc_type in ROUTES:
        out = {"route": ROUTES[doc_type], "reason": f"classified as {doc_type}"}
    else:
        out = {"route": "human", "reason": f"classified as {doc_type}, which no extractor supports yet"}
    return {**prediction, **out}
