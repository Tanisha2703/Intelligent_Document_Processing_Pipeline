"""
Export utilities — convert pipeline results to JSON, CSV, and Excel.

Functions:
  to_json(result)        → JSON string
  to_csv(result)         → CSV string (entities table)
  to_excel_bytes(result) → bytes (3 sheets: Summary, Entities, Raw Text)
"""
import io
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_json(result: dict, indent: int = 2) -> str:
    """
    Serialize pipeline result to a formatted JSON string.

    Adds a timestamp and removes large word_data list for cleaner output.
    """
    export = {
        "exported_at": datetime.now().isoformat(),
        "page_count": result.get("page_count", 1),
        "ocr_confidence": result.get("ocr_confidence", 0.0),
        "language": result.get("language", {}),
        "classification": result.get("classification", {}),
        "entity_count": result.get("entity_count", 0),
        "entities": result.get("entities", []),
        "entities_by_type": result.get("entities_by_type", {}),
        "processing_time_s": result.get("processing_time_s", 0.0),
        "ocr_text": result.get("ocr_text", ""),
    }
    return json.dumps(export, ensure_ascii=False, indent=indent)


def to_csv(result: dict) -> str:
    """
    Export entities to CSV format.

    Columns: entity_type, value, confidence, source
    """
    entities = result.get("entities", [])
    if not entities:
        return "entity_type,value,confidence,source\n"

    rows = []
    for ent in entities:
        rows.append(
            {
                "entity_type": ent.get("type", ""),
                "value": ent.get("value", ""),
                "confidence": ent.get("confidence", 0.0),
                "source": ent.get("source", ""),
            }
        )

    df = pd.DataFrame(rows, columns=["entity_type", "value", "confidence", "source"])
    return df.to_csv(index=False)


def to_excel_bytes(result: dict) -> bytes:
    """
    Export result as an Excel workbook with 3 sheets:

    Sheet 1 — Summary:
        Processing metadata (OCR confidence, language, classification, timing)

    Sheet 2 — Entities:
        Table of all extracted entities (type, value, confidence, source)

    Sheet 3 — Raw Text:
        Full OCR text in a single cell
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # ---- Sheet 1: Summary ----------------------------------------
        classification = result.get("classification", {})
        language = result.get("language", {})
        summary_data = {
            "Field": [
                "Export Timestamp",
                "Page Count",
                "OCR Confidence (%)",
                "Language (Primary)",
                "Hindi Ratio",
                "English Ratio",
                "Document Category",
                "Category Confidence",
                "Total Entities Found",
                "Processing Time (s)",
            ],
            "Value": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                result.get("page_count", 1),
                result.get("ocr_confidence", 0.0),
                language.get("primary", "unknown"),
                f"{language.get('hindi_ratio', 0.0):.1%}",
                f"{language.get('english_ratio', 0.0):.1%}",
                classification.get("category", "unknown"),
                f"{classification.get('confidence', 0.0):.1%}",
                result.get("entity_count", 0),
                result.get("processing_time_s", 0.0),
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # ---- Sheet 2: Entities ----------------------------------------
        entities = result.get("entities", [])
        if entities:
            rows = [
                {
                    "Entity Type": e.get("type", ""),
                    "Value": e.get("value", ""),
                    "Confidence": f"{e.get('confidence', 0.0):.1%}",
                    "Source": e.get("source", ""),
                    "Start (char)": e.get("start", ""),
                    "End (char)": e.get("end", ""),
                }
                for e in entities
            ]
            df_entities = pd.DataFrame(rows)
        else:
            df_entities = pd.DataFrame(
                columns=["Entity Type", "Value", "Confidence", "Source", "Start (char)", "End (char)"]
            )
        df_entities.to_excel(writer, sheet_name="Entities", index=False)

        # ---- Sheet 3: Raw Text ----------------------------------------
        raw_text = result.get("ocr_text", "")
        df_text = pd.DataFrame({"OCR Extracted Text": [raw_text]})
        df_text.to_excel(writer, sheet_name="Raw Text", index=False)

    return output.getvalue()
