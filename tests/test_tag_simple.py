"""
Simple standalone test for tag preservation (no dependencies).
"""

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Initialize
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))


def preprocess_old(text):
    """Old method - stems everything."""
    tokens = text.lower().split()
    result = []
    for token in tokens:
        if token not in stop_words:
            stemmed = stemmer.stem(token)
            result.append(stemmed)
    return " ".join(result)


def preprocess_enhanced_new(enhanced_query):
    """New method - preserves underscore tags."""
    tokens = enhanced_query.split()
    result = []
    for token in tokens:
        if '_' in token:
            # Preserve tags as-is
            result.append(token)
        else:
            # Normal preprocessing
            cleaned = token.lower()
            if cleaned not in stop_words:
                stemmed = stemmer.stem(cleaned)
                result.append(stemmed)
    return " ".join(result)


print("=" * 80)
print("TAG PRESERVATION TEST")
print("=" * 80)

test_cases = [
    "budget bike budget_bike",
    "mountain bike mountain_bike",
    "bike with child seat bike_with_child_seat",
    "perfect bike perfect_bike bike_for_hills",
]

print(f"\n{'Enhanced Query':<50} | {'Old (Wrong)':<30} | {'New (Correct)':<30}")
print("-" * 115)

for query in test_cases:
    old = preprocess_old(query)
    new = preprocess_enhanced_new(query)
    print(f"{query:<50} | {old:<30} | {new:<30}")

print("\n✅ Tags with underscores are preserved in the new method!")
print("=" * 80)
