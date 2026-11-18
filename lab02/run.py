"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import matplotlib
import torch
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
ATTENTION_SENTENCE = "Врач осмотрел пациента и сказал, что его анализы готовы."
ENGLISH_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. "
    "Nobody dared to leave before the host, and the host did not seem to notice the time at all. "
    "A servant brought tea, and the conversation turned, as it always did in that house, "
    "to the war and to the emperor."
)
SEARCH_QUERIES = [
    ("Хозяйка вечера несколько дней болела гриппом и утром разослала приглашения", "кашляла несколько дней"),
    ("Отец говорит, что оба его сына дураки, только один спокойный, а другой беспокойный", "покойный дурак"),
    ("Богатый и скупой старый князь живет в деревне, его прозвали прусским королем", "прозванный прусским королем"),
    ("Гости по очереди здоровались с никому не нужной тетушкой", "никому не интересной и не нужной тетушки"),
    ("Брат удивительно похож на сестру-красавицу, но при этом очень некрасив", "поразительно дурен собой"),
    ("В гостиную входит невысокий молодой князь с усталым и скучающим взглядом", "усталого, скучающего взгляда"),
    ("Обедневшая княгиня приехала просить, чтобы ее сына перевели в гвардию", "выхлопотать определение в гвардию"),
    ("Спор о революции: народ отдал Бонапарту власть, чтобы избавиться от Бурбонов", "избавил его от Бурбонов"),
    ("Неуклюжий толстый гость по рассеянности взял чужую генеральскую шляпу", "треугольную шляпу с генеральским плюмажем"),
    ("Мальчика десяти лет отправили за границу с гувернером-аббатом", "гувернером-аббатом за границу"),
    ("Совет другу: не женись, пока не сделал всего, что мог", "не женись до тех пор"),
    ("Светлая петербургская ночь, второй час, он едет домой и не может заснуть", "бессумрачная ночь"),
]


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

def extract_paragraphs(file_path: str | Path, max_paragraphs=500):
    paragraphs = []

    text = Path(file_path).read_text(encoding="utf-8")
    for paragraph in text.splitlines():
        normalized_paragraph = paragraph.lower().replace("ё", "е")
        words_list = re.findall(r"[а-я]+(?:-[а-я]+)*", normalized_paragraph)

        if 40 <= len(words_list) <= 120:
            paragraphs.append(paragraph)    

    return paragraphs[:max_paragraphs]    

def create_paragraphs_json(paragraphs):
    """Сохранить исходные абзацы с номерами в порядке отбора."""
    data = [{"id": i, "text": paragraph} for i, paragraph in enumerate(paragraphs)]
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "paragraphs.json"
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def get_static_vector(text: str, model, tokenizer):
    """Принимает текст, разбивает на токены, получает эмбэдинг каждого токена, 
        и усредняет их превращая в один вектор"""

    # Форма input_ids: [1, число токенов]. Каждый текст обрабатывается отдельно,
    # поэтому дополнения (padding) нет и все токены участвуют в среднем.
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    if input_ids.shape[1] == 0:
        raise ValueError("Нельзя получить средний вектор текста без токенов")

    embedding_layer = model.get_input_embeddings()
    input_ids = input_ids.to(embedding_layer.weight.device)
    with torch.inference_mode():
        # [1, число токенов, размерность вектора]. Слои внимания не вызываются.
        # Т.е. каждый токен имеет свой эмбэдинг, и мы их получаем
        token_vectors = embedding_layer(input_ids)
        # Усредняем по токенам: [1, размерность] -> [размерность].
        vector = token_vectors.float().mean(dim=1).squeeze(0)
    return vector.cpu()


def get_contextual_vector(text: str, model, tokenizer):
    """Пропустить текст через Qwen3 и усреднить состояния последнего слоя."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] == 0:
        raise ValueError("Нельзя получить средний вектор текста без токенов")

    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Как и у static: [1, число токенов, размерность], но уже после слоев модели.
        token_vectors = outputs.hidden_states[-1]
        # Обрабатываем один текст без padding, поэтому усредняем все его токены.
        vector = token_vectors.float().mean(dim=1).squeeze(0)
    return vector.cpu()


def get_e5_vector(text: str, model, tokenizer, prefix="passage"):
    """Получить вектор E5: префикс, энкодер, среднее по маске, нормализация."""
    if prefix not in ("query", "passage"):
        raise ValueError("Для E5 нужен префикс query или passage")
    inputs = tokenizer(
        f"{prefix}: {text}",
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)
    with torch.inference_mode():
        token_vectors = model(**inputs).last_hidden_state.float()
        # Маска равна 1 у настоящих токенов и 0 у padding.
        mask = inputs["attention_mask"].unsqueeze(-1).bool()
        token_vectors = token_vectors.masked_fill(~mask, 0.0)
        vector = token_vectors.sum(dim=1) / mask.sum(dim=1)
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)
    return vector.squeeze(0).cpu()


def search_by_cosine(query_vector, paragraph_vectors, k=5):
    """Выбрать номера k ближайших абзацев по готовым векторам любого способа."""
    # Сравниваем один вектор запроса с каждой строкой матрицы абзацев.
    similarities = torch.nn.functional.cosine_similarity(
        paragraph_vectors, query_vector.unsqueeze(0), dim=1,
    )
    # При одинаковой близости сохраняем исходный порядок абзацев.
    return torch.argsort(similarities, descending=True, stable=True)[:k].tolist()


def create_search_json(paragraphs, paragraph_vectors, model, tokenizer, e5_model=None, e5_tokenizer=None):
    """Идет по таблице запросов из задания, находит нужный абзац по ключевой фразе,
        получает вектор запроса, сравнивает этот вектор с каждым вектором абзацев, находит ближайшие"""
    k = 5
    queries = []
    # Для каждого способа передается своя матрица векторов абзацев.
    vector_functions = {
        "static": get_static_vector,
        "contextual": get_contextual_vector,
    }
    if "e5" in paragraph_vectors and (e5_model is None or e5_tokenizer is None):
        raise ValueError("Для поиска e5 нужно передать его модель и токенизатор")
    hits = {method: 0 for method in paragraph_vectors}
    for query, answer_phrase in SEARCH_QUERIES:

        # Находим id правильного абзаца (answer_id) по ключевой фразе answer_phrase
        normalized_phrase = answer_phrase.lower().replace("ё", "е")
        answer_id = next(
            (i for i, paragraph in enumerate(paragraphs)
             if normalized_phrase in paragraph.lower().replace("ё", "е")),
            None,
        )

        # Ищем k самых близких векторов абзацев к вектору запроса
        top = {}
        for method, vectors in paragraph_vectors.items():
            if method == "e5":
                query_vector = get_e5_vector(query, e5_model, e5_tokenizer, prefix="query")
            else:
                query_vector = vector_functions[method](query, model, tokenizer)
            top[method] = search_by_cosine(query_vector, vectors, k=k)
        
        # Определеяем, есть ли среди k самых близких векторов абзацев правильный абзац
        found = {
            method: answer_id is not None and answer_id in ids
            for method, ids in top.items()
        }

        # Общее количество попаданий
        for method in hits:
            hits[method] += int(found[method])
        
        queries.append({
            "query": query,
            "answer_phrase": answer_phrase,
            "answer_id": answer_id,
            "top": top,
        })
        print(
            f'Запрос {len(queries)}: top-{k}={top}, '
            f'правильный абзац={answer_id}, найден={found}'
        )

    # Если нужного абзаца нет в отобранном корпусе, запрос считается неуспешным.
    recall = {method: count / len(SEARCH_QUERIES) for method, count in hits.items()}
    data = {
        "k": k,
        "n_paragraphs": len(paragraphs),
        "queries": queries,
        "recall_at_k": recall,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "search.json"
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for method in hits:
        print(f'{method} recall@{k}: {hits[method]}/{len(SEARCH_QUERIES)} = {recall[method]:.3f}')
    return destination


def find_word_token_positions(sentence, word, offsets):
    """Найти токены, чьи символьные интервалы пересекают указанное слово."""
    # Находим границы первого целого слова в исходном тексте, а не его часть.
    # match.start() — начало слова, match.end() — позиция сразу после него.
    match = re.search(r"\b" + re.escape(word) + r"\b", sentence)
    if match is None:
        raise ValueError(f'Слово {word!r} отсутствует в предложении')
    # offsets[i] = (start, end) показывает, какие символы покрывает токен i.
    # Слово может состоять из нескольких токенов — собираем все пересечения.
    positions = [
        i for i, (start, end) in enumerate(offsets)
        if start < match.end() and end > match.start()
    ]
    if not positions:
        raise ValueError(f'Не найдены токены слова {word!r}')
    return positions


def create_attention_map(model, tokenizer, sentence=ATTENTION_SENTENCE):
    """Найти голову с максимальным вниманием от «его» к «пациента», сохранить JSON и график."""
    # Вместе с ID токенов получаем их границы в исходном предложении.
    inputs = tokenizer(
        sentence, return_tensors="pt", add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # Смещения нужны для поиска слов, но не являются входом модели.
    offsets = inputs.pop("offset_mapping")[0].tolist()
    pronoun_pos = find_word_token_positions(sentence, "его", offsets) # Местоположение "его"
    antecedent_pos = find_word_token_positions(sentence, "пациента", offsets) # Местоположение "пациента"
    # Читаемые подписи нужны для JSON и оси X графика; модель получает сами ID.
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    inputs = inputs.to(model.get_input_embeddings().weight.device)

    # Просим вернуть матрицы внимания всех слоев. Градиенты и кеш для
    # генерации не нужны: выполняем один проход по готовому предложению.
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    if outputs.attentions is None or any(a is None for a in outputs.attentions):
        raise ValueError('Для получения внимания загрузите Qwen3 с attn_implementation="eager"')

    # Любая доля внимания неотрицательна, поэтому первая голова станет лучшей.
    best_share = -1.0
    for layer, attention in enumerate(outputs.attentions):
        # attention: [1, головы, токены запроса, токены ключа].
        # Берем строку последнего токена местоимения для каждой головы.
        rows = attention[0, :, pronoun_pos[-1], :].float()
        # У слова «пациента» несколько токенов: складываем их веса отдельно
        # для каждой головы. Получаем по одной доле внимания на голову.
        shares = rows[:, antecedent_pos].sum(dim=1)
        # Сначала выбираем лучшую голову текущего слоя, затем сравниваем
        # ее с лучшими головами уже просмотренных слоев.
        head = int(shares.argmax().item())
        share = shares[head].item()
        if share > best_share:
            best_share = share
            best_layer = layer
            best_head = head
            # Сохраняем всю строку, чтобы показать внимание ко всем токенам.
            weights = rows[head].cpu().tolist()

    # Номера слоев, голов и токенов записываются с нуля.
    data = {
        "sentence": sentence,
        "tokens": tokens,
        "pronoun_pos": pronoun_pos,
        "antecedent_pos": antecedent_pos,
        "best_layer": best_layer,
        "best_head": best_head,
        "weights": weights,
        "share_on_antecedent": sum(weights[i] for i in antecedent_pos),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "attention_map.json"
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    # Высота столбика — вес внимания от «его» к соответствующему токену.
    # Оранжевым отмечаем «пациента», зеленым — токен, из которого смотрим.
    colors = ["tab:orange" if i in antecedent_pos else "tab:blue" for i in range(len(tokens))]
    colors[pronoun_pos[-1]] = "tab:green"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(tokens)), weights, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([f'{i}: {token!r}' for i, token in enumerate(tokens)], rotation=60, ha="right")
    ax.set_ylabel("Вес внимания от последнего токена «его»")
    ax.set_title(
        f'Слой {best_layer}, голова {best_head} (нумерация с 0); '
        f'доля на «пациента»: {data["share_on_antecedent"]:.1%}'
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "attention_map.png", dpi=150)
    # После сохранения освобождаем ресурсы фигуры.
    plt.close(fig)
    print(
        f'Внимание: слой {best_layer}, голова {best_head}, '
        f'доля на «пациента»: {data["share_on_antecedent"]:.4f}'
    )
    return destination


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
    ap.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu",
        help="устройство для моделей (по умолчанию cpu); auto выбирает CUDA, если она доступна",
    )
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA недоступна: используйте --device cpu или --device auto")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f'Устройство моделей: {device}')
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

    # Получаем список параграфов длиной от 40 до 120 слов для задания 3
    paragraphs = extract_paragraphs(corpus_destination, max_paragraphs=60 if args.fast else 500)
    print(f'Список парграфов длиной: {len(paragraphs)}')
    for i in range(len(paragraphs[:60])):
        print(f'{i}: {paragraphs[i]}')

    # Записываем параграфы в json для artifacts
    paragraphs_destination = create_paragraphs_json(paragraphs)
    print(f'Абзацы сохранены в {paragraphs_destination}')

    print('Загружаем Qwen3 для получения входных эмбеддингов')
    # На GPU используем float16; на CPU — float32 без медленной эмуляции bf16.
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        TOKENIZER_NAME, dtype=model_dtype, attn_implementation="eager",
    ).to(device)
    model.eval()
    tokenizer = get_qwen_tokenizer()

    # Берем полученные абзацы и переводим каждый абзац в один вектор
    # static_vectors это список векторов, каждый из которых представляет собой один абзац
    static_vectors = torch.stack([
        get_static_vector(paragraph, model, tokenizer)
        for paragraph in paragraphs
    ])
    print(f'Форма матрицы static: {tuple(static_vectors.shape)}')
    print(f'Первые 5 чисел вектора первого абзаца: {static_vectors[0, :5].tolist()}')

    # Контекстный метод - переводим параграфы в токены, пропускает через внимание модели
    # берем выходные векторы
    contextual_vectors = []
    for i, paragraph in enumerate(paragraphs):
        contextual_vectors.append(get_contextual_vector(paragraph, model, tokenizer))
        print(f'\rContextual: {i + 1}/{len(paragraphs)} абзацев', end="", flush=True)
    contextual_vectors = torch.stack(contextual_vectors)
    print(f'\nФорма матрицы contextual: {tuple(contextual_vectors.shape)}')

    print('Загружаем E5 для поиска по смыслу')
    e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    e5_model = AutoModel.from_pretrained(E5_MODEL_NAME).to(device)
    e5_model.eval()
    e5_vectors = []
    for i, paragraph in enumerate(paragraphs):
        e5_vectors.append(get_e5_vector(paragraph, e5_model, e5_tokenizer, prefix="passage"))
        print(f'\rE5: {i + 1}/{len(paragraphs)} абзацев', end="", flush=True)
    e5_vectors = torch.stack(e5_vectors)
    print(f'\nФорма матрицы e5: {tuple(e5_vectors.shape)}')

    search_destination = create_search_json(
        paragraphs,
        {"static": static_vectors, "contextual": contextual_vectors, "e5": e5_vectors},
        model,
        tokenizer,
        e5_model=e5_model,
        e5_tokenizer=e5_tokenizer,
    )
    print(f'Результаты static, contextual и e5 сохранены в {search_destination}')

    # Задание 4а: внимание местоимения к антецеденту.
    attention_destination = create_attention_map(model, tokenizer)
    print(f'Карта внимания сохранена в {attention_destination} и attention_map.png')

if __name__ == "__main__":
    main()
