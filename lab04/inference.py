import re
import torch


def parse_rating(answer):
    # Ищем в answer первую цифру от 1 до 5, которая не является частью другого числа
    match = re.search(r"(?<!\d)[1-5](?!\d)", answer)
    return int(match.group()) if match else None


@torch.inference_mode()
def predict(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    output = model.generate(
        **inputs,
        do_sample=False, # Выключение случайного выбора. Используется жадная генерация
        max_new_tokens=3
    )

    # generate возвращает промпт вместе с ответом. Оставляем только ответ.
    prompt_length = inputs["input_ids"].shape[1]
    answer_tokens = output[0, prompt_length:]
    raw_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return raw_answer, parse_rating(raw_answer)
