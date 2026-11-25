"""Лабораторная 4. Задание: assignments/04/README.md в репозитории курса.

Запуск: python lab04/run.py, для CI python lab04/run.py --fast.
Адаптер обучается в lab04/train.ipynb и лежит в lab04/artifacts/adapter/.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

from data import DATASET_NAME, count_classes, prepare_data

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()

    train_data, few_shot_data, test_data = prepare_data(
        load_dataset(DATASET_NAME), fast=args.fast,
    )
    class_counts = count_classes(test_data)
    ARTIFACTS.mkdir(exist_ok=True)
    metrics = {
        "dataset": DATASET_NAME,
        "n_test": len(test_data),
        "fast": args.fast,
        "class_counts": class_counts,
        "modes": {},
    }
    metrics_path = ARTIFACTS / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train: {len(train_data)}, few-shot: {len(few_shot_data)}, test: {len(test_data)}")
    print(f"Распределение теста (звёзды): {class_counts}")
    print(f"Сохранено: {metrics_path}")


if __name__ == "__main__":
    main()
