import re
import string

from nltk.stem import WordNetLemmatizer


def preprocess(text: str) -> str:

    text = remove_punctuation(text)

    text = lower_text(text)

    text = remove_emoji(text)

    text = convert_num_to_label(text)

    lemmatizer = WordNetLemmatizer()
    text = lemmatize_text(text, lemmatizer)

    return text


def remove_punctuation(text: str) -> str:
    return "".join([c for c in text if c not in string.punctuation])


def lower_text(text: str) -> str:
    return text.lower()


def remove_emoji(string: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def convert_num_to_label(text: str) -> str:
    num_pattern = re.compile("[0-9]+")
    return num_pattern.sub(r"number", text)


def lemmatize_text(
    text: str, lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
) -> str:
    text = text.lower()
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())
