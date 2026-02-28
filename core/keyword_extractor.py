from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(text, top_n=15):
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=200,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9+#._-]{1,}\b'
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        keyword_scores = [
            {"keyword": feature_names[i], "score": round(float(scores[i]), 4)}
            for i in range(len(feature_names))
            if scores[i] > 0
        ]
        keyword_scores.sort(key=lambda x: x["score"], reverse=True)
        return keyword_scores[:top_n]
    except Exception:
        return []


def compare_keywords(cv_keywords, jd_keywords):
    cv_set = {kw["keyword"].lower() for kw in cv_keywords}
    jd_set = {kw["keyword"].lower() for kw in jd_keywords}

    matched = sorted(cv_set & jd_set)
    missing = sorted(jd_set - cv_set)

    if len(jd_set) > 0:
        similarity_score = round(len(matched) / len(jd_set), 2)
    else:
        similarity_score = 0.0

    return {
        "matched": matched,
        "missing": missing,
        "similarity_score": similarity_score
    }
