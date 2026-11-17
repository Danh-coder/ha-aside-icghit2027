"""ChatQA text conversations: application-owned roles, unchanged source strings."""

from experiments.prompts import Chunk, PREFACE, render_chunks, encode_chunks
from ha_aside.core import ROLE_SYSTEM, ROLE_USER, ROLE_DATA, ROLE_ASSISTANT

RECIPE = "chatqa_multirole_fullseq_2048_v1"
DATASET_NAME = "chatqa_multirole_v1"
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_LENGTH = 2048
ROLE_POLICY = "application preface/framing=SYSTEM; document=DATA; each user=USER; assistant history/answer=ASSISTANT"


def validate_conversation(row):
    if not isinstance(row.get("document"), str) or not row["document"].strip():
        raise ValueError("ChatQA document must be nonempty text")
    messages, answers = row.get("messages"), row.get("answers")
    if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
        raise ValueError("ChatQA history must end with a user question")
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if (message.get("role") != expected or not isinstance(message.get("content"), str)
                or not message["content"].strip()):
            raise ValueError("ChatQA requires nonempty alternating user/assistant messages")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str) or not answers[0].strip():
        raise ValueError("ChatQA requires exactly one unchanged text answer")


def conversation_chunks(row, tokenizer):
    validate_conversation(row)
    messages = [
        ("system", [Chunk(PREFACE, ROLE_SYSTEM)]),
        ("user", [Chunk("Document:\n", ROLE_SYSTEM), Chunk(row["document"], ROLE_DATA, True),
                  Chunk("\n\nQuestion:\n", ROLE_SYSTEM), Chunk(row["messages"][0]["content"], ROLE_USER)]),
    ]
    for message in row["messages"][1:]:
        role = ROLE_USER if message["role"] == "user" else ROLE_ASSISTANT
        messages.append((message["role"], [Chunk(message["content"], role)]))
    messages.append(("assistant", [Chunk(row["answers"][0], ROLE_ASSISTANT, target=True)]))
    return render_chunks(tokenizer, messages, generation_prompt=False)


def encode_conversation(row, tokenizer, max_length=MAX_LENGTH):
    return encode_chunks(tokenizer, conversation_chunks(row, tokenizer), max_length)
