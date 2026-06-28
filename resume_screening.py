
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("Resume.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ===============================
# Clean Text Function
# ===============================

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

df['Resume_str'] = df['Resume_str'].apply(clean_text)

# ===============================
# Job Description
# ===============================

job_description = """
Python
Machine Learning
Data Analysis
SQL
Pandas
Scikit Learn
Communication Skills
"""

job_description = clean_text(job_description)

# ===============================
# TF-IDF Vectorization
# ===============================

documents = df['Resume_str'].tolist()
documents.append(job_description)

tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(documents)

# ===============================
# Similarity Score
# ===============================

similarity = cosine_similarity(
    matrix[-1],
    matrix[:-1]
)

df['Score'] = similarity[0] * 100

# ===============================
# Ranking Candidates
# ===============================

df = df.sort_values(
    by='Score',
    ascending=False
)

print("\nTop Candidates\n")

print(df[['Category', 'Score']].head(10))

# ===============================
# Missing Skills
# ===============================

required_skills = {
    'python',
    'machine learning',
    'sql',
    'data analysis',
    'pandas',
    'scikit learn'
}

def missing_skills(resume):
    resume = resume.lower()
    missing = []

    for skill in required_skills:
        if skill not in resume:
            missing.append(skill)

    return ", ".join(missing)

df['Missing Skills'] = df['Resume_str'].apply(
    missing_skills
)

print("\nTop Ranked Candidates\n")
print(
    df[['Category',
        'Score',
        'Missing Skills']].head(10)
)

# ===============================
# Save Results
# ===============================

df.to_csv(
    "ranked_candidates.csv",
    index=False
)

print("\nResults saved successfully.")

import matplotlib.pyplot as plt

top = df.head(10)

plt.figure(figsize=(10,6))
plt.bar(
    top['Category'],
    top['Score']
)

plt.xticks(rotation=90)
plt.xlabel("Candidate Category")
plt.ylabel("Matching Score")
plt.title("Top Candidate Ranking")
plt.tight_layout()
plt.show()