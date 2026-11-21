"""Лабораторная 3. Задание: assignments/03/README.md в репозитории курса.

Запуск: python lab03/run.py, для CI python lab03/run.py --fast.
Слияния BPE берутся из lab02/artifacts/bpe.json.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen
import re

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
BPE = HERE.parent / "lab02" / "artifacts" / "bpe.json"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"


def read_json(file_path: str | Path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def download_file(url: str) -> Path:
    """Скачать файл по ссылке в corpus.txt рядом со скриптом и вернуть путь."""
    destination = HERE / "corpus.txt"
    with urlopen(url, timeout=30) as response:
        content = response.read()
    destination.write_bytes(content)
    return destination

def extract_words(file_path: str | Path) -> list[str]:
    text = Path(file_path).read_text(encoding="utf-8")
    normalized_text = text.lower().replace("ё", "е")
    words_list = re.findall(r"[а-я]+(?:-[а-я]+)*", normalized_text)
    return words_list

def create_words_dict(words):
    """Убирает слова, встречаемость которых ниже порога"""
    words_count = {}
    words_dict = {}

    for word in words:
        words_count[word] = words_count.get(word, 0) + 1

    for word, repetitions in words_count.items():
        words_dict[word] = repetitions

    return words_dict

def split_chars(words_count):
    """Каждый ключ разбивает на символы"""
    words_dict = {}
    for word, frequency in words_count.items():
        words_dict[tuple(word) + ('</w>',)] = frequency
    return words_dict

def replace_chars(chars, key):
    new_chars = []
    i = 0
    while i < len(chars)-1:
        if (chars[i], chars[i+1]) != key:
            new_chars.append(chars[i])
            i += 1
        else:
            new_chars.append(key[0]+key[1])
            i += 2
    # Добавляем последний символ
    if i < len(chars):
        new_chars.append(chars[i])
    return new_chars

def tokenize(word_dict, merges):
    tokens = word_dict.copy()
    for i in range(len(merges)):
        key = merges[i]
        new_tokens = {}
        for chars, frequency in tokens.items():
            merged = tuple(replace_chars(chars, key))
            new_tokens[merged] = new_tokens.get(merged, 0) + frequency
        tokens = new_tokens

        print(f"\rСлияние {i + 1} из {len(merges)}, пара: {key}", end="", flush=True)

    print()
    return tokens


def words_to_ids(words, tokens, token_to_id):
    word_to_tokens = {}
    for parts in tokens:
        word = "".join(parts).removesuffix("</w>")
        word_to_tokens[word] = parts

    # Исходный список сохраняет порядок и повторы слов, в отличие от словаря.
    ids = []
    for word in words:
        for token in word_to_tokens[word]:
            ids.append(token_to_id[token])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    # Скачиваем текст
    corpus_destination = download_file(CORPUS_URL)
    print(f'Файл скачан в {corpus_destination}')

    # Обрабатываем текст, получаем список слов
    words = extract_words(corpus_destination)
    words_train = words[:int(len(words)*0.9)]  # Первые 90%
    words_test = words[int(len(words)*0.9):]  # Последние 10%
    print(f'Текст прочитан, количество слов в train: {len(words_train)}, в test: {len(words_test)}')

    # Формируем словарь с полученным списком слов
    words_dict_train = split_chars(create_words_dict(words_train))
    words_dict_test = split_chars(create_words_dict(words_test))

    # Получаем список слияний из прошлой лабораторной
    merges = [tuple(pair) for pair in read_json(BPE)['merges']]
    print(f'Количество слияний: {len(merges)}')

    # Разбиваем текст на токены
    tokens_train = tokenize(words_dict_train, merges)
    tokens_test = tokenize(words_dict_test, merges)

    # Собираем токены, реально встретившиеся в корпусе после применения BPE.
    # Для train и val используем один общий словарь и одинаковые ID.
    vocab = set()
    for parts in tokens_train:
        vocab.update(parts)
    for parts in tokens_test:
        vocab.update(parts)

    # Сортируем, чтобы номера токенов не менялись при повторных запусках.
    token_to_id = {token: i for i, token in enumerate(sorted(vocab))}
    id_to_token = {i: token for token, i in token_to_id.items()}

    train_ids = words_to_ids(words_train, tokens_train, token_to_id)
    val_ids = words_to_ids(words_test, tokens_test, token_to_id)
    print(f'Размер словаря BPE: {len(token_to_id)}')
    print(f'Токенов в train: {len(train_ids)}, в val: {len(val_ids)}')


if __name__ == "__main__":
    main()
