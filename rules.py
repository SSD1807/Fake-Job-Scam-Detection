def rule_based_score(row):
    score = 0

    if row.get("has_company_logo", 0) == 0:
        score += 10

    if row.get("has_questions", 0) == 0:
        score += 10

    text = row.get("combined_text", "")

    if "fee" in text or "registration" in text or "payment" in text:
        score += 30

    if "whatsapp" in text or "telegram" in text:
        score += 20

    if "guaranteed" in text:
        score += 15

    if "work from home" in text:
        score += 5

    return score