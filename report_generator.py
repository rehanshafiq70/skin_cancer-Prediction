from datetime import datetime

def generate_report(result, confidence):

    report = f"""
SkinScan AI Report
---------------------
Diagnosis : {result}
Confidence : {confidence:.2f} %

Generated : {datetime.now()}
"""

    return report
