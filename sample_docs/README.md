# Sample Documents

Three test documents are provided for demonstrating the pipeline. Use these by dragging them into the Streamlit app.

## Generate Sample Documents

Run the script below to create all three:

```bash
python sample_docs/generate_samples.py
```

This creates:
- `sample_docs/hindi_invoice.png` — Hindi invoice
- `sample_docs/english_receipt.png` — English payment receipt
- `sample_docs/mixed_form.png` — Mixed Hindi/English government form

---

## Generation Script

Create `sample_docs/generate_samples.py`:

```python
"""
Generate sample test documents using Pillow.
Run: python sample_docs/generate_samples.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


def make_image(lines: list[str], width: int = 800, line_height: int = 40,
               bg: str = "white", fg: str = "black") -> Image.Image:
    """Create a white image with lines of text."""
    height = len(lines) * line_height + 80
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill=fg)
        y += line_height
    return img


# ---- Document 1: Hindi Invoice ----------------------------------------
hindi_invoice_lines = [
    "कर चालान / Tax Invoice",
    "─────────────────────────────────────────────",
    "चालान संख्या: INV-00123",
    "दिनांक: 15/03/2024",
    "GSTIN: 27ABCDE1234F1Z5",
    "",
    "प्रेषक: टेकसॉफ्ट सॉल्यूशंस प्राइवेट लिमिटेड",
    "पता: 45, एमजी रोड, पुणे, महाराष्ट्र",
    "",
    "प्राप्तकर्ता: श्री राजेश कुमार शर्मा",
    "पैन संख्या: ABCDE1234F",
    "आधार: 1234 5678 9012",
    "फोन: +91 9876543210",
    "ईमेल: rajesh.sharma@example.com",
    "",
    "विवरण                     राशि",
    "─────────────────────────────────────────────",
    "परामर्श सेवाएं              ₹40,000",
    "CGST 9%                    ₹3,600",
    "SGST 9%                    ₹3,600",
    "─────────────────────────────────────────────",
    "कुल देय राशि               ₹47,200",
    "",
    "बैंक विवरण: HDFC Bank, A/C: 12345678901234",
    "IFSC: HDFC0001234",
]
img1 = make_image(hindi_invoice_lines)
img1.save(OUTPUT_DIR / "hindi_invoice.png", dpi=(300, 300))
print("Created: sample_docs/hindi_invoice.png")


# ---- Document 2: English Receipt ----------------------------------------
english_receipt_lines = [
    "PAYMENT RECEIPT",
    "═════════════════════════════════════════════",
    "Receipt No.: RCP-2024-0456",
    "Date: 22 March 2024",
    "",
    "Received from:  Priya Sharma",
    "PAN:            FGHIJ5678K",
    "Phone:          +91 8765432109",
    "Email:          priya.sharma@gmail.com",
    "",
    "─────────────────────────────────────────────",
    "Description            Amount",
    "─────────────────────────────────────────────",
    "Annual Subscription    Rs. 12,000",
    "GST (18%)              Rs.  2,160",
    "─────────────────────────────────────────────",
    "Total Paid             Rs. 14,160",
    "─────────────────────────────────────────────",
    "",
    "Mode of Payment: UPI (GPay)",
    "Reference No.: UPI20240322PRIYA456",
    "",
    "For: Acme Technologies Pvt. Ltd.",
    "     Mumbai, Maharashtra",
    "     GSTIN: 27FGHIJ5678K1Z2",
    "",
    "Authorized Signatory: _________________",
    "                      Ravi Mehta (CFO)",
]
img2 = make_image(english_receipt_lines)
img2.save(OUTPUT_DIR / "english_receipt.png", dpi=(300, 300))
print("Created: sample_docs/english_receipt.png")


# ---- Document 3: Mixed Hindi/English Form ----------------------------------------
mixed_form_lines = [
    "भारत सरकार / Government of India",
    "आयकर विभाग / Income Tax Department",
    "═════════════════════════════════════════════",
    "Form 16 — Certificate of Tax Deduction",
    "Assessment Year: 2024-25",
    "",
    "नियोक्ता का नाम / Employer Name:",
    "  InnovateTech Solutions Private Limited",
    "TAN: DELH12345A",
    "GSTIN: 07ABCDE1234F1Z5",
    "पता / Address: B-42, Connaught Place, New Delhi",
    "",
    "कर्मचारी का नाम / Employee Name: Amir Khan",
    "PAN / पैन: AMRKH1234J",
    "आधार संख्या / Aadhaar No.: 9876 5432 1098",
    "Designation / पद: Senior Software Engineer",
    "",
    "Period / अवधि: 01/04/2023 to 31/03/2024",
    "",
    "वित्तीय विवरण / Financial Details:",
    "  Gross Salary:          Rs. 12,00,000",
    "  HRA Exemption:         Rs.  1,80,000",
    "  Standard Deduction:    Rs.     50,000",
    "  Total Tax Deducted:    Rs.  1,20,000",
    "",
    "Employee Phone: +91 7654321098",
    "Email: amir.khan@innovatetech.in",
    "",
    "Signature / हस्ताक्षर: _______________",
    "Date / दिनांक: 15/04/2024",
]
img3 = make_image(mixed_form_lines)
img3.save(OUTPUT_DIR / "mixed_form.png", dpi=(300, 300))
print("Created: sample_docs/mixed_form.png")

print("\nAll sample documents generated!")
print("Drag any of these into the Streamlit app to test the pipeline.")
```

---

## What to Expect

### Hindi Invoice
- **Language**: Hindi (primary)
- **Category**: Invoice
- **Expected entities**: PERSON (राजेश कुमार शर्मा), ORG (टेकसॉफ्ट सॉल्यूशंस), AMOUNT (₹47,200), DATE (15/03/2024), ID_NUMBER (ABCDE1234F, 1234 5678 9012), PHONE (+91 9876543210), EMAIL (rajesh.sharma@example.com)

### English Receipt
- **Language**: English (primary)
- **Category**: Receipt
- **Expected entities**: PERSON (Priya Sharma), ORG (Acme Technologies), AMOUNT (Rs. 14,160), DATE (22 March 2024), ID_NUMBER (FGHIJ5678K, RCP-2024-0456), PHONE (+91 8765432109), EMAIL (priya.sharma@gmail.com)

### Mixed Form
- **Language**: Mixed (Hindi + English)
- **Category**: Govt Form
- **Expected entities**: PERSON (Amir Khan), ORG (InnovateTech Solutions), AMOUNT (Rs. 12,00,000), DATE (01/04/2023), ID_NUMBER (AMRKH1234J, 9876 5432 1098), PHONE (+91 7654321098), EMAIL (amir.khan@innovatetech.in)
