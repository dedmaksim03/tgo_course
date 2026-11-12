"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
ENGLISH_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. "
    "Nobody dared to leave before the host, and the host did not seem to notice the time at all. "
    "A servant brought tea, and the conversation turned, as it always did in that house, "
    "to the war and to the emperor."
)


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
    merges = []

    for i in range(num_merges):
        key, count = find_max_repetitions(tokens)
        if key is None:
            break
        new_tokens = {}
        for chars, frequency in tokens.items():
            merged = tuple(replace_chars(chars, key))
            new_tokens[merged] = new_tokens.get(merged, 0) + frequency
        tokens = new_tokens
        merges.append(key)

        print(f"\rСлияние {i + 1} из {num_merges}, пара: {key}", end="", flush=True)
    return tokens, merges

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

    return tokens


def create_bpe_json(stats, merges, words_dict):

    # Полный словарь BPE состоит из всех изначальных токенов + новые токены на каждом шаге merges
    vocab = {char for chars in words_dict for char in chars}
    vocab.update(left + right for left, right in merges)

    examples = {}
    for word in ("наполеон", "сказала", "неудовольствие"):
        chars = list(word) + ["</w>"]
        for key in merges:
            chars = replace_chars(chars, key)
        examples[word] = chars

    data = {
        "num_merges": len(merges),
        "vocab_size": len(vocab),
        "merges": merges,
        "stats": stats,
        "examples": examples,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "bpe.json"
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
            
@lru_cache(maxsize=1)
def get_qwen_tokenizer():
    """Загрузить токенизатор один раз; веса модели здесь не нужны."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Число токенов Qwen3 в тексте без добавления специальных токенов."""
    tokenizer = get_qwen_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def compare_tokens_per_word(words):
    """Сравнить русский и английский отрывки и сохранить tokens_per_word.json."""
    tokenizer = get_qwen_tokenizer()
    # Сто слов нормализованного корпуса: срез с индексами 500–599.
    russian_paragraph = " ".join(words[500:600])
    context_tokens = 4096
    data = {
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": len(tokenizer),
        "context_tokens": context_tokens,
    }
    for language, text in (("ru", russian_paragraph), ("en", ENGLISH_PARAGRAPH)):
        word_count = len(text.split())
        token_count = count_tokens(text)
        tokens_per_word = token_count / word_count
        data[language] = {
            "words": word_count,
            "tokens": token_count,
            "tokens_per_word": tokens_per_word,
            "words_in_context": int(context_tokens / tokens_per_word),
        }
        print(
            f'{language}: {word_count} слов, {token_count} токенов, '
            f'{tokens_per_word:.3f} токена на слово'
        )

    # .encode возвращает закодированный список токенов, на которые он разбил текст
    # каждый токен это число - и .decode позволяет получить токен по этому числу
    russian_ids = tokenizer.encode(russian_paragraph, add_special_tokens=False)
    data["example_ru"] = [tokenizer.decode([token_id]) for token_id in russian_ids]
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "tokens_per_word.json"
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


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
    print(f'Текст прочитан, количество всех слов: {len(words)}')

    # Убираем слова ниже порога
    cleaned_words_train = delete_words_below_threshold(words, th=5)
    cleaned_words_all = delete_words_below_threshold(words, th=1)
    print(f'После фильтрации: {sum(cleaned_words_train.values())} слов, уникальных: {len(cleaned_words_train)}')

    # Разбиваем ключ на символы, добавляем маркер конца строки
    words_dict_train = split_chars(cleaned_words_train)
    words_dict_all = split_chars(cleaned_words_all)
    initial_len_train = sum(len(chars) * frequency for chars, frequency in words_dict_train.items())
    initial_len_all = sum(len(chars) * frequency for chars, frequency in words_dict_all.items())
    print(f'Количество символов с маркерами в обучающем корпусе/во всем корпусе: \
          {initial_len_train}/{initial_len_all}')

    max_key, max_value = find_max_repetitions(words_dict_train)
    print(f'Самая часто встречающаяся пара {max_key}, {max_value}')

    # Запускаем обучение BPE для получения токенов и слияний
    _, merges = bpe(words_dict_train, num_merges=200 if args.fast else 1500)

    # Запускаем токенизацию на всем корпусе, включая редкие слова
    tokens = tokenize(words_dict_all, merges)

    print("\n")
    for word_tokens, frequency in list(tokens.items())[:10]:
        print(f"Токены: {word_tokens}, частота: {frequency}")

    # Считаем количество символов и количество слов во всем корпусе
    count_chars = sum(
        (len(chars) - 1) * frequency
        for chars, frequency in words_dict_all.items()
    )
    count_words = sum(words_dict_all.values())
    count_tokens = sum(
        len(t) * frequency
        for t, frequency in tokens.items()        
    )

    print(f'Количество символов/слов/токенов: {count_chars}/{count_words}/{count_tokens}')

    stats = {
        "chars": count_chars,
        "words": count_words,
        "tokens_bpe": count_tokens,
        "tokens_per_word": count_tokens / count_words,
        "compression_vs_chars": count_chars / count_tokens,
    }

    # Задание 1: создаем bpe.json
    bpe_destination = create_bpe_json(stats, merges, words_dict_all)
    print(f'Результаты BPE сохранены в {bpe_destination}')

    # Задание 2: используем токенизатор Qwen и сравниваем число токенов для русского и английского текста
    tokens_destination = compare_tokens_per_word(words)
    print(f'Сравнение токенизации сохранено в {tokens_destination}')


    #raise NotImplementedError("работа не начата")

if __name__ == "__main__":
    main()
