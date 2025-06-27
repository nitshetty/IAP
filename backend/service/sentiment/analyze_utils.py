from sqlalchemy.orm import Session
from db.models import SentimentLabel

def calculate_percentages(label_match_counts):
    total = sum(label_match_counts.values())
    if total == 0:
        return {k: 0 for k in label_match_counts}
    perc = {k: (v * 100.0 / total) for k, v in label_match_counts.items()}
    perc_rounded = {k: int(round(v)) for k, v in perc.items()}
    diff = 100 - sum(perc_rounded.values())
    if diff != 0:
        main_key = max(perc, key=perc.get)
        perc_rounded[main_key] += diff
    return perc_rounded

def analyze_sentiment_with_percentage(text: str, db: Session):
    label_data = db.query(SentimentLabel).all()
    print(f"[DEBUG] Sentiment labels from DB: {label_data}")
    label_match_counts = {label.sentiment.lower(): 0 for label in label_data}
    print(f"[DEBUG] Initial label match counts: {label_match_counts}")
    total_keywords = 0
    print(f"[DEBUG] Input text: {repr(text)}")
    for label_entry in label_data:
        keywords = [kw.strip().lower() for kw in label_entry.keyword.split(',')]
        print(f"[DEBUG] Checking label '{label_entry.sentiment}' with keywords: {keywords}")
        for keyword in keywords:
            if keyword in text.lower():
                print(f"[DEBUG] Found keyword match: '{keyword}' in text")
                label_match_counts[label_entry.sentiment.lower()] += 1
                total_keywords += 1
    print(f"[DEBUG] Final label match counts: {label_match_counts}")
    print(f"[DEBUG] Total keywords matched: {total_keywords}")
    if total_keywords == 0:
        print("[DEBUG] No keywords matched. Returning Neutral.")
        return {"summary": "Neutral", "percentage": {}}
    # Calculate float percentages for accuracy
    perc_floats = {label: (count / total_keywords) * 100 for label, count in label_match_counts.items() if count > 0}
    # Round to int, then fix rounding error
    perc_rounded = {label: int(round(val)) for label, val in perc_floats.items()}
    diff = 100 - sum(perc_rounded.values())
    if diff != 0 and perc_rounded:
        # Add/subtract the difference to the label with the highest float value
        main_label = max(perc_floats, key=perc_floats.get)
        perc_rounded[main_label] += diff
    print(f"[DEBUG] Percentages: {perc_rounded}")
    summary = max(perc_rounded, key=perc_rounded.get)
    print(f"[DEBUG] Summary: {summary}")
    return {"summary": summary.capitalize(), "percentage": perc_rounded}
