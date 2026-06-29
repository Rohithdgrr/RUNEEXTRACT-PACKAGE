"""Fine-tuning data generation — create training examples from documents for LLM fine-tuning."""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FineTuneExample:
    instruction: str
    input: str = ""
    output: str = ""
    context: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


@dataclass
class FineTuneDataset:
    examples: List[FineTuneExample] = field(default_factory=list)
    format: str = "alpaca"

    def add(self, example: FineTuneExample):
        self.examples.append(example)

    def save(self, path: str, fmt: Optional[str] = None):
        fmt = fmt or self.format
        data = [e.to_dict() for e in self.examples]
        if fmt == "sharegpt":
            conversations = []
            for e in self.examples:
                conv = {
                    "conversations": [
                        {"from": "human", "value": e.input or e.instruction},
                        {"from": "gpt", "value": e.output},
                    ]
                }
                conversations.append(conv)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
        elif fmt == "messages":
            messages_list = []
            for e in self.examples:
                msgs = [
                    {"role": "system", "content": e.instruction},
                ]
                if e.input:
                    msgs.append({"role": "user", "content": e.input})
                else:
                    msgs.append({"role": "user", "content": e.instruction})
                msgs.append({"role": "assistant", "content": e.output})
                messages_list.append({"messages": msgs})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages_list, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    @property
    def size(self) -> int:
        return len(self.examples)


QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Summarize the following passage about {topic}.",
    "Explain the key points about {topic}.",
    "What are the main findings related to {topic}?",
    "Describe {topic} in detail.",
    "List the important aspects of {topic}.",
    "What does the document say about {topic}?",
    "Provide an overview of {topic}.",
    "What are the conclusions about {topic}?",
    "Extract the key information about {topic}.",
]

SUMMARIZE_TEMPLATES = [
    "Summarize the following text in 2-3 sentences.",
    "What is the main idea of this passage?",
    "Provide a concise summary of the key points.",
    "What are the most important takeaways from this text?",
]


def extract_topics(text: str, max_topics: int = 5) -> List[str]:
    if not text or len(text.strip()) < 30:
        return []
    import re
    sentences = re.split(r'[.!?\n]+', text)
    topics = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        words = [w for w in sent.split() if len(w) > 4 and w[0].isupper()]
        if words:
            topic = " ".join(words[:3])
            if topic not in topics:
                topics.append(topic)
        if len(topics) >= max_topics:
            break
    return topics or ["document content"]


def generate_examples_from_document(
    doc: Any,
    num_examples: int = 5,
    include_summaries: bool = True,
    seed: int = 42,
) -> FineTuneDataset:
    rng = random.Random(seed)
    dataset = FineTuneDataset()
    text = getattr(doc, "text", "")
    if not text or len(text) < 50:
        return dataset
    topics = extract_topics(text, max_topics=num_examples + 2)
    used_templates = rng.sample(QUESTION_TEMPLATES, min(num_examples, len(QUESTION_TEMPLATES)))
    for i in range(num_examples):
        if i < len(used_templates):
            topic = rng.choice(topics) if topics else "this document"
            instruction = used_templates[i].format(topic=topic)
        else:
            topic = rng.choice(topics) if topics else "document content"
            instruction = f"Answer a question about {topic}."
        paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 50]
        if not paragraphs:
            paragraphs = [text[:1000]]
        context = rng.choice(paragraphs)
        output = _generate_answer_from_context(instruction, context)
        dataset.add(FineTuneExample(
            instruction=instruction,
            input=context[:500],
            output=output,
            context=context,
            source=getattr(doc, "source_path", "") or "",
            metadata={"document_id": getattr(doc, "document_id", "")},
        ))
    if include_summaries:
        paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 100]
        for para in paragraphs[:3]:
            template = rng.choice(SUMMARIZE_TEMPLATES)
            dataset.add(FineTuneExample(
                instruction=template,
                input=para[:500],
                output=para[:200],
                context=para,
                source=getattr(doc, "source_path", "") or "",
                metadata={"type": "summary"},
            ))
    return dataset


def _generate_answer_from_context(instruction: str, context: str) -> str:
    import re
    topic_match = re.search(r"(?:about|of|related to)\s+(.+?)(?:\?|\.)", instruction)
    if topic_match:
        topic = topic_match.group(1).strip()
        sentences = re.split(r'[.!?\n]+', context)
        relevant = [s.strip() for s in sentences if topic.lower() in s.lower() and len(s.strip()) > 10]
        if relevant:
            return relevant[0][:300]
    return context[:300].strip()


def generate_fine_tuning_data(
    docs: List[Any],
    examples_per_doc: int = 5,
    output_path: Optional[str] = None,
    format: str = "alpaca",
) -> FineTuneDataset:
    dataset = FineTuneDataset(format=format)
    for i, doc in enumerate(docs):
        doc_dataset = generate_examples_from_document(doc, num_examples=examples_per_doc, seed=i)
        for ex in doc_dataset.examples:
            dataset.add(ex)
    if output_path:
        dataset.save(output_path)
    return dataset
