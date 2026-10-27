"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    # Скачиваем текст
    corpus_destination = download_file(CORPUS_URL)
    print(f'Файл скачан в {corpus_destination}')

    raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
