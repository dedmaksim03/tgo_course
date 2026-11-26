MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "Определи оценку, которую автор поставил организации, по тексту отзыва "
    "по шкале от 1 до 5. Где 1 - плохо, а 5 - хорошо. Ответь одной цифрой от 1 до 5"
)


def build_messages(review, examples=()):
    # Записываем промты под разные роли. Модель получает эти промты со специальными токенами
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.extend([
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": str(example["stars"])},
        ])
    messages.append({"role": "user", "content": review})
    return messages


def format_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
