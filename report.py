from datetime import datetime

def generate_report(label,confidence,risk):

    return f"""
SkinScan AI Medical Report

Diagnosis: {label}
Confidence: {confidence:.2f}%
Risk Level: {risk}

Recommendation:
Consult dermatologist if needed.

Generated:
{datetime.now()}
"""
