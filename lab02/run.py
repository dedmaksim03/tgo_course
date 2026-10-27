"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
import re
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"


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

def split_chars(words_list: list[str]) -> list[str]:
    chars_list = []
    for word in words_list: 
        for char in word:
            chars_list.append(char)
        chars_list.append('</w>') # Добавляем маркер конца слова
    
    return chars_list

def find_max_repetitions(chars):
    """Ищет самую частую пару символов"""
    pairs = {}
    max_key = None 
    max_value = 0
    for i in range(len(chars)-1):
        if chars[i] == '</w>': # Пара не должна начинаться с конца слова
            continue 
        current_key = (chars[i], chars[i+1])
        current_value = pairs.get(current_key, 0)

        pairs[current_key] = current_value + 1

        if pairs.get(current_key) > max_value:
            max_key = current_key
            max_value = pairs.get(current_key)

    return max_key, pairs.get(max_key, 0)

def delete_words_below_threshold(words, th=2):
    """Убирает слова, встречаемость которых ниже порога"""
    words_count = {}
    cleaned_words = []

    for word in words:
        words_count[word] = words_count.get(word, 0) + 1

    for word in words:
        repetitions = words_count[word]

        if repetitions >= th:
            cleaned_words.append(word)

    return cleaned_words

def bpe(chars):
    pass

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
    print(f'Текст прочитан, количество слов: {len(words)}')

    # Убираем слова ниже порога
    cleaned_words = delete_words_below_threshold(words, th=2)
    print(f'Слова очищены, кол-во слов было {len(words)}, стало {len(cleaned_words)}')

    # Разбиваем на символы, добавляем маркер конца строки
    chars_list = split_chars(cleaned_words)
    print(f'Текст разбит на символы {len(chars_list)}')

    max_key, max_value = find_max_repetitions(chars_list)
    print(f'Самая часто встречающаяся пара {max_key}, {max_value}')

    #raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
