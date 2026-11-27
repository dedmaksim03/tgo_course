import torch
from prompts import build_messages, format_prompt


def prepare_example(example, tokenizer, max_len=256):
    # Нужны для расчета loss
    answer_ids = tokenizer.encode(
        str(example["stars"]) + tokenizer.eos_token,
        add_special_tokens=False, # Они будут добавлены в format_prompt
    )
    # Нужны для обрезки промта до нужной длины
    review_ids = tokenizer.encode(example["text"], add_special_tokens=False)

    review = example["text"]

    # Полный промт должен поместиться в max_len
    # В случае overflow обрезаем сам отзыв и проверяем снова
    while True:
        prompt = format_prompt(tokenizer, build_messages(review))
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        overflow = len(prompt_ids) + len(answer_ids) - max_len
        if overflow <= 0:
            break

        # Обрезаем отзыв до нужной длины
        review_ids = review_ids[:max(0, len(review_ids) - overflow)]
        review = tokenizer.decode(review_ids, skip_special_tokens=True)

    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def collate_batch(examples, pad_token_id):
    # Собирает примеры в батч, накладывая маску, определяющую настоящие токены и pad_token
    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_mask, labels = [], [], []
    for example in examples:
        length = len(example["input_ids"])
        padding = max_len - length
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * length + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
