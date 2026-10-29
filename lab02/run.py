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

def split_chars(words_count):
    """Каждый ключ разбивает на символы"""
    words_dict = {}
    for word, frequency in words_count.items():
        words_dict[tuple(word) + ('</w>',)] = frequency
    return words_dict

def find_max_repetitions(words_dict):
    """Ищет самую частую пару символов"""
    pairs = {}
    max_key = None 
    max_value = 0
    for chars, frequency in words_dict.items():
        for i in range(len(chars)-1):
            current_key = (chars[i], chars[i+1])
            current_value = pairs.get(current_key, 0) + frequency
            pairs[current_key] = current_value

            if current_value > max_value:
                max_key = current_key
                max_value = current_value

    return max_key, pairs.get(max_key, 0)

def delete_words_below_threshold(words, th=2):
    """Убирает слова, встречаемость которых ниже порога"""
    words_count = {}
    cleaned_words = {}

    for word in words:
        words_count[word] = words_count.get(word, 0) + 1

    for word, repetitions in words_count.items():
        if repetitions >= th:
            cleaned_words[word] = repetitions

    return cleaned_words

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

def bpe(words_dict, num_merges=1500):

    tokens = words_dict.copy()

    for i in range(num_merges):
        key, count = find_max_repetitions(tokens)
        if key is None:
            break
        new_tokens = {}
        for chars, frequency in tokens.items():
            merged = tuple(replace_chars(chars, key))
            new_tokens[merged] = new_tokens.get(merged, 0) + frequency
        tokens = new_tokens

        print(f"\rСлияние {i + 1} из {num_merges}, пара: {key}", end="", flush=True)
    return tokens
            


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
    print(f'После фильтрации: {sum(cleaned_words.values())} слов, уникальных: {len(cleaned_words)}')

    # Разбиваем ключ на символы, добавляем маркер конца строки
    words_dict = split_chars(cleaned_words)
    initial_len = sum(len(chars) * frequency for chars, frequency in words_dict.items())
    print(f'Количество символов с маркерами в обучающем корпусе: {initial_len}')

    max_key, max_value = find_max_repetitions(words_dict)
    print(f'Самая часто встречающаяся пара {max_key}, {max_value}')

    # Запускаем метод BPE для получения токенов
    tokens = bpe(words_dict, num_merges=200 if args.fast else 1500)
    final_len = sum(len(chars) * frequency for chars, frequency in tokens.items())
    print(f'Исходная длина/итоговая длина обучающего корпуса: {initial_len}/{final_len}')
    for word_tokens, frequency in list(tokens.items())[:15]:
        print(f"Токены: {word_tokens}, частота: {frequency}")
    #raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
