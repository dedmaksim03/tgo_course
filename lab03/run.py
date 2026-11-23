"""Лабораторная 3. Задание: assignments/03/README.md в репозитории курса.

Запуск: python lab03/run.py, для CI python lab03/run.py --fast.
Слияния BPE берутся из lab02/artifacts/bpe.json.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen
import re

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
BPE = HERE.parent / "lab02" / "artifacts" / "bpe.json"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"


class TokenPositionEmbeddings(nn.Module):
    """Превратить ID токенов в векторы и добавить информацию об их позициях."""

    def __init__(self, vocab_size, d_model=128, context=128):
        super().__init__()
        self.context = context
        # Создаем эмбэдинги токенов по размеру словаря
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Создаем эмбэдинги позиций по размеру контекста
        self.position_embedding = nn.Embedding(context, d_model)

    def forward(self, input_ids):
        # Ожидаем целочисленный тензор [число окон в батче, длина окна].
        if input_ids.ndim != 2:
            raise ValueError("input_ids должен иметь форму [batch, length]")
        length = input_ids.shape[1]
        if length > self.context:
            raise ValueError(f'Длина окна {length} превышает контекст {self.context}')

        # Позиции начинаются с нуля в каждом окне: [0, 1, ..., length - 1].
        positions = torch.arange(length, device=input_ids.device)
        token_vectors = self.token_embedding(input_ids)  # [batch, length, d_model]
        position_vectors = self.position_embedding(positions)  # [length, d_model]
        # PyTorch добавит одни и те же позиционные векторы ко всем окнам батча.
        return token_vectors + position_vectors


class CausalSelfAttention(nn.Module):
    """Многоголовое внимание: каждый токен видит только себя и предыдущие."""

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model должен быть положительным и делиться на n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Каждая проекция читает весь входной вектор. Ее выход затем
        # разделяется на головы: у каждой будут свои Q, K и V.
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch, length, d_model], например [1, 128, 128].
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x должен иметь форму [batch, length, d_model]")
        batch, length, _ = x.shape

        # [batch, length, 128] -> [batch, length, 4, 32] -> [batch, 4, length, 32].
        # Переносим ось голов, чтобы считать внимание всех голов параллельно.
        q = self.query(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        # Оценки каждой пары токенов: [batch, heads, length, length].
        # Делим на корень из размерности одной головы, как в формуле лекции.
        # Строка — токен, из которого смотрим, столбец — токен, на который смотрим.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Запрещаем первому токену видеть второй
        # True выше диагонали означает «будущий токен». Ставим -inf ДО
        # softmax, чтобы его вес стал нулевым, а доступные веса дали сумму 1.
        future_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))

        # Оценки scores — произвольные числа: отрицательные, положительные, большие или маленькие. 
        # Softmax превращает их в неотрицательные веса с суммой 1
        weights = torch.softmax(scores, dim=-1)

        # Взвешенная сумма V: [batch, heads, length, head_dim].
        attended = weights @ v
        # Возвращаем головы на соседние оси и склеиваем их координаты.
        # contiguous() делает данные подходящими для последующего view().
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.output(attended)


class DecoderBlock(nn.Module):
    """Один блок декодера: нормализация, внимание, FFN и две остаточные связи."""

    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        # Одна и та же сеть применяется к каждому токену отдельно.
        # Расширяем вектор 128 -> 512, добавляем нелинейность, сжимаем до 128.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # Pre-norm: сначала нормализуем вход ветки, потом считаем поправку.
        # К исходному x добавляем результат внимания, сохраняя прежний сигнал.
        x = x + self.attention(self.norm1(x))
        # FFN добавляет еще одну поправку, не смешивая разные позиции.
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLanguageModel(nn.Module):
    """Эмбеддинги, несколько блоков декодера и оценки следующего токена."""

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=512, context=128):
        super().__init__()
        if n_layers <= 0 or d_ff <= 0:
            raise ValueError("Число слоев и ширина FFN должны быть положительными")
        self.context = context
        # Сохраняем настройки, с которыми действительно создается модель.
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "context": context,
        }
        self.embeddings = TokenPositionEmbeddings(vocab_size, d_model=d_model, context=context)
        # Создаем отдельные блоки: у каждого свои веса внимания, FFN и нормализаций.
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)  # [batch, length, d_model]
        for block in self.blocks:
            # Выход предыдущего блока становится входом следующего.
            x = block(x)
        x = self.final_norm(x)
        # Для каждой позиции выдаем по одной оценке на каждый токен словаря.
        # Это логиты, не вероятности: softmax здесь не применяем.
        return self.output(x)  # [batch, length, vocab_size]


def get_batch(token_ids, batch_size, context=128, generator=None):
    """Взять случайные окна и правильные ответы, сдвинутые на один токен."""
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)

    # starts это список начала окон, размером с batch_size
    starts = torch.randint(len(token_ids) - context, (batch_size,), generator=generator).tolist()
    inputs = torch.stack([token_ids[start:start + context] for start in starts])

    # Формируем targets как inputs со сдвигом
    # Предсказывать будем не последний 129ый токен, а для каждого токена следующий после него
    targets = torch.stack([token_ids[start + 1:start + context + 1] for start in starts])
    return inputs, targets


def calculate_loss(model, inputs, targets):
    """Посчитать среднюю ошибку предсказания следующего токена по всему батчу."""
    logits = model(inputs)  # [batch, context, vocab_size]

    # Размер logits: [batch, context, vocab_size]
    # Размер targets: [batch, context]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def save_model_config(model):
    """Записать архитектуру, точное число параметров и устройство модели."""
    config = model.config.copy()
    # Считаем реальные тензоры параметров, а не приблизительный размер модели.
    config["n_params"] = sum(parameter.numel() for parameter in model.parameters())
    config["device"] = str(next(model.parameters()).device)
    ARTIFACTS.mkdir(exist_ok=True)
    destination = ARTIFACTS / "tiny_config.json"
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return destination


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
    ap.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="устройство для модели: cpu (по умолчанию) или cuda",
    )
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("GPU с поддержкой CUDA недоступен; используйте --device cpu")
    device = torch.device(args.device)
    print(f'Устройство модели: {device}')
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

    # Первое окно обучающего потока: пока один пример в батче.
    torch.manual_seed(42)
    model = TinyLanguageModel(vocab_size=len(token_to_id)).to(device)
    embedding_layer = model.embeddings
    input_ids = torch.tensor(
        train_ids[:embedding_layer.context], dtype=torch.long, device=device,
    ).unsqueeze(0)
    vectors = embedding_layer(input_ids)
    print(f'Размеры окна: {tuple(input_ids.shape)}')
    print(f'Размер итоговых векторов (позиция + токены): {tuple(vectors.shape)}')

    attention_layer = model.blocks[0].attention
    contextual_vectors = attention_layer(vectors)
    print(f'Размер векторов после внимания: {tuple(contextual_vectors.shape)}')

    logits = model(input_ids)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f'Размер логитов после четырех блоков декодера: {tuple(logits.shape)}')
    print(f'Число параметров модели: {n_params}')
    config_destination = save_model_config(model)
    print(f'Конфигурация модели сохранена в {config_destination}')

    # Потоки и получаемые из них батчи находятся на том же устройстве, что модель.
    train_data = torch.tensor(train_ids, dtype=torch.long, device=device)
    val_data = torch.tensor(val_ids, dtype=torch.long, device=device)
    batch_size = 16 if args.fast else 32
    train_generator = torch.Generator().manual_seed(42)
    train_inputs, train_targets = get_batch(
        train_data, batch_size=batch_size, context=model.context, generator=train_generator,
    )

    # Отдельный генератор не зависит от случайного выбора обучающих окон.
    # Эти 20 окон создаем один раз и будем повторно использовать для проверки.
    val_generator = torch.Generator().manual_seed(43)
    val_inputs, val_targets = get_batch(
        val_data, batch_size=20, context=model.context, generator=val_generator,
    )

    # Пока только измеряем начальную ошибку: веса здесь не обновляются.
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss(model, train_inputs, train_targets).item()
        val_loss = calculate_loss(model, val_inputs, val_targets).item()
    model.train()
    print(f'Размер обучающего батча: {tuple(train_inputs.shape)}')
    print(f'Размер фиксированного проверочного батча: {tuple(val_inputs.shape)}')
    print(f'Начальный train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')


if __name__ == "__main__":
    main()
