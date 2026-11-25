"""Общие выборки для обучения адаптера и оценки трёх режимов."""

import random
from collections import Counter

DATASET_NAME = "mteb/GeoreviewClassification"
SEED = 42


def count_classes(split) -> dict[str, int]:
    """Возвращает количество примеров по каждой оценке в виде словаря"""
    counts = Counter(split["stars"])
    return {str(stars): counts[stars] for stars in range(1, 6)}


def sample_per_class(split, count: int, seed: int):
    """Выбирает из датасета (выборки split) часть данных (count примеров)"""

    # Сначала создаем словарь с отзывами
    # Ключ - класс, значение - номер (индекс) отзыва с этим классом
    indices_by_label = {label: [] for label in range(5)}
    for index, label in enumerate(split["label"]):
        if label not in indices_by_label:
            raise ValueError(f"Неожиданная метка {label}: ожидались 0–4")
        indices_by_label[label].append(index)

    rng = random.Random(seed)
    indices = []
    for label, candidates in indices_by_label.items():
        indices.extend(rng.sample(candidates, count))
    rng.shuffle(indices)
    
    # .select возвращает новый датасет, содержащий только выбранные примеры
    return split.select(indices) 


def with_stars(split):
    # Добавляет колонку "stars" - количество звезд, т.к. label 0-4, а stars должен быть 1-5
    # label остаётся исходным (0–4), stars используется в ответах и метриках.
    return split.add_column("stars", [int(label) + 1 for label in split["label"]])


def prepare_data(dataset, seed = SEED, fast = False):
    n_test = 24 if fast else 1000
    if len(dataset["test"]) < n_test:
        raise ValueError(f"Для оценки требуется минимум {n_test} тестовых примеров")
    return (
        with_stars(sample_per_class(dataset["train"], 1000, seed)), # 1000 примеров на каждый класс
        with_stars(sample_per_class(dataset["validation"], 1, seed)), # 1 пример на каждый класс
        with_stars(dataset["test"].select(range(n_test))), # Выбираем n_test примеров в исходном порядке
    )
