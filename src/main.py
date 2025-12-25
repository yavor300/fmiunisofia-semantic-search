import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Сваляне на stop words, ако липсват
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class ProductSearchEngine:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))

    def load_data(self, file_path, limit=1000):
        """
        Зарежда данни от ЛОКАЛЕН файл.
        """
        print(f"Зареждане на данни от файл: {file_path}...")

        if not os.path.exists(file_path):
            print(f"ГРЕШКА: Файлът не е намерен на този път: {file_path}")
            return

        try:
            # Четем CSV-то
            self.df = pd.read_csv(file_path, on_bad_lines="skip")
            self.df = self.df.head(limit)

            # --- Мапинг на колоните според вашата схема ---
            # Вашите колони: title, final_price, description, brand

            # Създаваме вътрешни унифицирани колони
            # Използваме .get(), за да не гърми, ако някоя липсва, но очакваме да ги има
            self.df["name"] = self.df["title"].fillna("")
            self.df["brand"] = self.df["brand"].fillna("")
            self.df["desc"] = self.df["description"].fillna("")
            self.df["price_raw"] = self.df["final_price"].fillna(0)

            # Изчисляваме числова цена за семантичното обогатяване
            self.df["price_numeric"] = self.df["price_raw"].apply(self._clean_price)

            # Филтрираме продукти без име
            self.df = self.df[self.df["name"] != ""]

            print(f"Успешно заредени {len(self.df)} продукта.")
            print("Примерни данни:", self.df[["name", "price_numeric"]].head(1).values)

        except Exception as e:
            print(f"Грешка при зареждане: {e}")

    def _clean_price(self, price_str):
        if isinstance(price_str, (int, float)):
            return float(price_str)
        # Regex за извличане на числа от текст като "$23.99" или "23,99"
        found = re.findall(r"[-+]?\d*\.\d+|\d+", str(price_str))
        if found:
            return float(found[0])
        return 0.0

    def preprocess_text(self, text):
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = text.split()
        tokens = [
            self.stemmer.stem(word) for word in tokens if word not in self.stop_words
        ]
        return " ".join(tokens)

    def semantic_enrichment(self):
        """Задача 3: Добавяне на тагове (budget, premium) спрямо цената"""
        print("Прилагане на семантично обогатяване...")
        enriched_data = []

        for _index, row in self.df.iterrows():
            tags = []
            price = row["price_numeric"]

            # [cite_start]Правила за цени (Задача 3 от презентацията) [cite: 113, 114, 115]
            if 0 < price < 30:
                tags.append("budget affordable cheap low-cost")
            elif price > 150:
                tags.append("premium expensive high-end luxury")

            # Комбинираме: Име + Марка + Описание + Тагове
            content = f"{row['name']} {row['brand']} {row['desc']} {' '.join(tags)}"

            processed = self.preprocess_text(content)
            enriched_data.append(processed)

        self.df["processed_content"] = enriched_data
        # Чистим празните
        self.df = self.df[self.df["processed_content"].str.len() > 0]

    def build_index(self):
        """Задача 4: Индексиране с TF-IDF [cite: 58, 59]"""
        print("Изграждане на индекс (TF-IDF)...")
        if self.df.empty:
            print("Няма данни за индексиране.")
            return

        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_content"])
        print(f"Индексът е готов. Размерност: {self.tfidf_matrix.shape}")

    def search(self, query, top_k=5):
        """Задача 9: Търсене с косинусова близост [cite: 161, 168]"""
        if self.vectorizer is None:
            return []

        processed_query = self.preprocess_text(query)
        if not processed_query:
            return []

        query_vec = self.vectorizer.transform([processed_query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_k:-1]

        results = []
        for i in related_docs_indices:
            if cosine_similarities[i] > 0:
                item = self.df.iloc[i]
                results.append(
                    {
                        "Product": item["name"],
                        "Brand": item["brand"],
                        "Price": item["price_raw"],  # Показваме оригиналната цена
                        "Score": round(cosine_similarities[i], 3),
                    }
                )
        return results


if __name__ == "__main__":
    # --- НАСТРОЙКА НА ПЪТЯ ---
    # Тъй като показахте, че файлът е в друга папка, слагаме абсолютен път
    # или относителен спрямо мястото, откъдето пускате скрипта.

    # Вариант 1: Абсолютен път (най-сигурно)
    csv_path = "/mnt/c/Users/Yavor/Downloads/Sofia-University-Facultet-of-Mathematcs-and-Informatics/Year-1/Semester-1/Information-Retrieval/eCommerce-dataset-samples/amazon-products.csv"

    # Вариант 2: Ако копирате файла при main.py, използвайте просто:
    # csv_path = "amazon-products.csv"

    engine = ProductSearchEngine()

    # 1. Зареждане
    engine.load_data(csv_path, limit=1000)

    # 2. Обогатяване и Индексиране
    if engine.df is not None and not engine.df.empty:
        engine.semantic_enrichment()
        engine.build_index()

        # 3. Търсене
        test_query = "cheap running shoes"
        print(f"\n--- Търсене за: '{test_query}' ---")
        results = engine.search(test_query)

        if not results:
            print("Няма намерени резултати.")

        for r in results:
            # Отрязваме името, ако е твърде дълго, за да се чете лесно в конзолата
            short_name = (
                (r["Product"][:60] + "..") if len(r["Product"]) > 60 else r["Product"]
            )
            print(f"[{r['Score']}] {short_name} | {r['Brand']} | {r['Price']}")
