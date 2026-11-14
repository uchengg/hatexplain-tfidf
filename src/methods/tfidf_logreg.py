import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_predict(train_df, test_df):
    """
    Train this method on train_df and return predictions for test_df.
    """
    required_train = {"text", "label"}
    missing_train = required_train - set(train_df.columns)
    if missing_train:
        raise ValueError(f"train_df is missing required columns: {sorted(missing_train)}")

    if "text" not in test_df.columns:
        raise ValueError("test_df is missing required column: 'text'")

    if len(train_df) == 0:
        raise ValueError("train_df has no rows; cannot fit TF-IDF or logistic regression.")

    if len(test_df) == 0:
        return np.array([], dtype=int)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    if X_train.shape[1] == 0:
        raise ValueError(
            "TF-IDF produced zero features; training data may be too small for min_df=2."
        )
    y_train = train_df["label"].to_numpy()

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    X_test = vectorizer.transform(test_df["text"].astype(str))
    y_pred = clf.predict(X_test)

    return np.asarray(y_pred, dtype=int)
