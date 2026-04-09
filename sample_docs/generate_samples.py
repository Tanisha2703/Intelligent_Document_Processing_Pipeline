"""
Generate sample test documents using Pillow.

Usage:
    python sample_docs/generate_samples.py

Creates:
    sample_docs/hindi_invoice.png
    sample_docs/english_receipt.png
    sample_docs/mixed_form.png
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).parent


def make_image(
    lines: list,
    width: int = 800,
    line_height: int = 36,
    bg: str = "white",
    fg: str = "black",
    padding: int = 60,
) -> Image.Image:
    """Create a crisp white image with evenly spaced text lines."""
    height = len(lines) * line_height + padding * 2
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((40, y), line, fill=fg)
        y += line_height
    return img


# ---- Document 1: Hindi Invoice ----------------------------------------
hindi_invoice_lines = [
    "         कर चालान / Tax Invoice",
    "=========================================",
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
    "-----------------------------------------",
    "परामर्श सेवाएं            Rs. 40,000",
    "CGST 9%                   Rs.  3,600",
    "SGST 9%                   Rs.  3,600",
    "-----------------------------------------",
    "कुल देय राशि              Rs. 47,200",
    "",
    "बैंक: HDFC Bank, A/C: 12345678901234",
    "IFSC: HDFC0001234",
]

img1 = make_image(hindi_invoice_lines)
out1 = OUTPUT_DIR / "hindi_invoice.png"
img1.save(out1, dpi=(300, 300))
print(f"Created: {out1}")


# ---- Document 2: English Receipt ----------------------------------------
english_receipt_lines = [
    "         PAYMENT RECEIPT",
    "=========================================",
    "Receipt No.: RCP-2024-0456",
    "Date: 22 March 2024",
    "",
    "Received from:  Priya Sharma",
    "PAN:            FGHIJ5678K",
    "Phone:          +91 8765432109",
    "Email:          priya.sharma@gmail.com",
    "",
    "Description            Amount",
    "-----------------------------------------",
    "Annual Subscription    Rs. 12,000",
    "GST (18%)              Rs.  2,160",
    "-----------------------------------------",
    "Total Paid             Rs. 14,160",
    "",
    "Mode of Payment: UPI (GPay)",
    "Reference No.: UPI20240322PRIYA456",
    "",
    "For: Acme Technologies Pvt. Ltd.",
    "     Mumbai, Maharashtra",
    "     GSTIN: 27FGHIJ5678K1Z2",
    "",
    "Authorized Signatory: Ravi Mehta (CFO)",
]

img2 = make_image(english_receipt_lines)
out2 = OUTPUT_DIR / "english_receipt.png"
img2.save(out2, dpi=(300, 300))
print(f"Created: {out2}")


# ---- Document 3: Mixed Hindi/English Form ----------------------------------------
mixed_form_lines = [
    "  भारत सरकार / Government of India",
    "  आयकर विभाग / Income Tax Department",
    "=========================================",
    "Form 16 -- Certificate of Tax Deduction",
    "Assessment Year: 2024-25",
    "",
    "Employer / नियोक्ता:",
    "  InnovateTech Solutions Private Limited",
    "  TAN: DELH12345A",
    "  Address: B-42, Connaught Place, New Delhi",
    "",
    "Employee / कर्मचारी: Amir Khan",
    "PAN / पैन: AMRKH1234J",
    "Aadhaar / आधार: 9876 5432 1098",
    "Designation: Senior Software Engineer",
    "",
    "Period: 01/04/2023 to 31/03/2024",
    "",
    "Financial Details / वित्तीय विवरण:",
    "  Gross Salary:       Rs. 12,00,000",
    "  HRA Exemption:      Rs.  1,80,000",
    "  Standard Deduction: Rs.     50,000",
    "  Total Tax Deducted: Rs.  1,20,000",
    "",
    "Phone: +91 7654321098",
    "Email: amir.khan@innovatetech.in",
    "",
    "Signature: _______________",
    "Date / दिनांक: 15/04/2024",
]

img3 = make_image(mixed_form_lines)
out3 = OUTPUT_DIR / "mixed_form.png"
img3.save(out3, dpi=(300, 300))
print(f"Created: {out3}")

print("\nAll 3 sample documents generated!")
print("Drag any of them into the Streamlit app (streamlit run app.py) to test.")
