import re


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)   # emails
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_documents(docs):
    cleaned_docs = [clean_text(doc) for doc in docs if doc.strip()]
    return cleaned_docs