import json


class HallucinationDatasetLoader:
    def _dataset_api(self):
        from datasets import load_dataset, Dataset, DatasetDict
        return load_dataset, Dataset, DatasetDict

    def _text(self, x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return "\n".join(self._text(v) for v in x)
        if isinstance(x, dict):
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    def _first_nonempty(self, values):
        for v in values:
            t = self._text(v).strip()
            if t:
                return t
        return ""

    def _first_faitheval_document(self, context):
        if isinstance(context, dict):
            return self._text(context.get("Document", ""))
        if isinstance(context, str):
            parts = context.split("Document:")
            if len(parts) >= 2:
                return parts[1].strip()
            return context
        return self._text(context)

    def _hotpot_reference(self, context):
        if not isinstance(context, dict):
            return self._text(context)
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        chunks = []
        for title, sent in zip(titles, sentences):
            sent_text = " ".join(self._text(s) for s in sent) if isinstance(sent, list) else self._text(sent)
            chunks.append(f"{self._text(title)}\n{sent_text}".strip())
        return "\n\n".join(chunks)

    def _twowiki_reference(self, context):
        if not isinstance(context, list):
            return self._text(context)
        chunks = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = self._text(item[0])
                body = " ".join(self._text(v) for v in item[1]) if isinstance(item[1], list) else self._text(item[1])
                chunks.append(f"{title}\n{body}".strip())
            else:
                chunks.append(self._text(item))
        return "\n\n".join(chunks)

    def _extract_drop_answer(self, row):
        spans = row.get("answers_spans", {})
        if isinstance(spans, dict):
            span_list = spans.get("spans", [])
            if isinstance(span_list, list) and span_list:
                return self._text(span_list[0]).strip()
        answers = row.get("answers", {})
        if isinstance(answers, dict):
            for key in ("spans", "number", "date"):
                value = answers.get(key)
                if isinstance(value, list) and value:
                    return self._text(value[0]).strip()
                if isinstance(value, dict):
                    composed = " ".join(self._text(v).strip() for v in value.values() if self._text(v).strip())
                    if composed:
                        return composed
                txt = self._text(value).strip()
                if txt:
                    return txt
        return ""

    def _extract_halueval_qa_answer(self, row):
        return self._first_nonempty([
            row.get("right_answer"),
            row.get("answer"),
            row.get("output"),
        ])

    def _extract_halueval_qa_question(self, row):
        return self._first_nonempty([
            row.get("question"),
            row.get("query"),
        ])

    def _extract_halueval_qa_reference(self, row):
        return self._first_nonempty([
            row.get("knowledge"),
            row.get("context"),
        ])

    def _extract_halueval_dialogue_question(self, row):
        return self._first_nonempty([
            row.get("dialogue_history"),
            row.get("query"),
            row.get("question"),
        ])

    def _extract_halueval_dialogue_reference(self, row):
        return self._first_nonempty([
            row.get("knowledge"),
            row.get("context"),
        ])

    def _extract_halueval_dialogue_answer(self, row):
        return self._first_nonempty([
            row.get("right_response"),
            row.get("output"),
            row.get("answer"),
        ])

    def load_by_name(self, name):
        fn = getattr(self, f"load_{name}", None)
        if fn is None:
            raise ValueError(f"Unsupported dataset name: {name}")
        return fn()

    def load_anah(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("opencompass/anah")
        rows = [
            {
                "id": self._text(r.get("name", "")),
                "question": self._text(r.get("selected_questions", "")),
                "answer": self._first_nonempty([r.get("answer"), r.get("output")]),
                "reference": self._text(r.get("documents", "")),
            }
            for r in ds["train"]
        ]
        return DatasetDict({"train": Dataset.from_list(rows)})

    def load_faitheval_inconsistent(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("Salesforce/FaithEval-inconsistent-v1.0")
        out = {}
        for split in ds.keys():
            rows = []
            for row in ds[split]:
                answers = row.get("answers")
                answer = self._text(answers[0]) if isinstance(answers, list) and len(answers) > 0 else self._text(answers)
                rows.append({
                    "id": self._text(row.get("qid", "")),
                    "question": self._text(row.get("question", "")),
                    "answer": answer,
                    "reference": self._first_faitheval_document(row.get("context", "")),
                })
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_truthfulqa(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("domenicrosati/TruthfulQA")
        rows = [
            {
                "id": self._text(r.get("Question", "")),
                "question": self._text(r.get("Question", "")),
                "answer": self._text(r.get("Best Answer", "")),
                "reference": "",
            }
            for r in ds["train"]
        ]
        return DatasetDict({"train": Dataset.from_list(rows)})

    def load_ragtruth(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("wandb/RAGTruth-processed")
        rows = [
            {
                "id": self._text(r.get("id", "")),
                "question": self._text(r.get("query", "")),
                "answer": self._text(r.get("output", "")),
                "reference": self._text(r.get("context", "")),
            }
            for r in ds["train"]
        ]
        return DatasetDict({"train": Dataset.from_list(rows)})

    def load_halueval_qa(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("pminervini/HaluEval", "qa")
        out = {}
        for split in ds.keys():
            rows = []
            for r in ds[split]:
                rows.append({
                    "id": self._first_nonempty([r.get("id"), r.get("question"), r.get("query")]),
                    "question": self._extract_halueval_qa_question(r),
                    "answer": self._extract_halueval_qa_answer(r),
                    "reference": self._extract_halueval_qa_reference(r),
                })
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_halueval_dialogue(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("pminervini/HaluEval", "dialogue")
        out = {}
        for split in ds.keys():
            rows = []
            for i, r in enumerate(ds[split]):
                rows.append({
                    "id": self._first_nonempty([r.get("id"), f"dialogue-{split}-{i}"]),
                    "question": self._extract_halueval_dialogue_question(r),
                    "answer": self._extract_halueval_dialogue_answer(r),
                    "reference": self._extract_halueval_dialogue_reference(r),
                })
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_halueval_summarization(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("pminervini/HaluEval", "summarization")
        out = {}
        for split in ds.keys():
            rows = []
            for i, r in enumerate(ds[split]):
                rows.append({
                    "id": self._first_nonempty([r.get("id"), f"summarization-{split}-{i}"]),
                    "question": "Summarize the document.",
                    "answer": self._first_nonempty([r.get("right_summary"), r.get("summary")]),
                    "reference": self._first_nonempty([r.get("document"), r.get("context")]),
                })
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_drop(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("ucinlp/drop")
        out = {}
        for split in ds.keys():
            rows = []
            for r in ds[split]:
                rows.append({
                    "id": self._text(r.get("query_id", "")),
                    "question": self._text(r.get("question", "")),
                    "answer": self._extract_drop_answer(r),
                    "reference": self._text(r.get("passage", "")),
                })
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_hotpotqa_fullwiki(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        out = {}
        for split in ds.keys():
            rows = [{
                "id": self._text(r.get("id", "")),
                "question": self._text(r.get("question", "")),
                "answer": self._text(r.get("answer", "")),
                "reference": self._hotpot_reference(r.get("context", {})),
            } for r in ds[split]]
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)

    def load_twowikimultihopqa(self):
        load_dataset, Dataset, DatasetDict = self._dataset_api()
        ds = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
        out = {}
        for split in ds.keys():
            rows = [{
                "id": self._text(r.get("_id", "")),
                "question": self._text(r.get("question", "")),
                "answer": self._text(r.get("answer", "")),
                "reference": self._twowiki_reference(r.get("context", [])),
            } for r in ds[split]]
            out[split] = Dataset.from_list(rows)
        return DatasetDict(out)
