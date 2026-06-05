from datasets import load_dataset, Dataset, DatasetDict
import json


class HallucinationDatasetLoader:
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

    def _first_faitheval_document(self, context):
        if isinstance(context, dict):
            return self._text(context.get("Document", ""))
        if isinstance(context, str):
            parts = context.split("Document:")
            if len(parts) >= 3:
                return parts[1].strip()
            if len(parts) == 2:
                return parts[1].strip()
            return context
        return self._text(context)

    def _hotpot_reference(self, context):
        if not isinstance(context, dict):
            return self._text(context)

        titles = context.get("title", [])
        sentences = context.get("sentences", [])

        chunks = []
        n = min(len(titles), len(sentences))
        for i in range(n):
            title = self._text(titles[i])
            sent = sentences[i]
            if isinstance(sent, list):
                sent = " ".join(self._text(s) for s in sent)
            else:
                sent = self._text(sent)
            chunks.append(f"{title}\n{sent}".strip())

        return "\n\n".join(chunks)

    def _twowiki_reference(self, context):
        if not isinstance(context, list):
            return self._text(context)

        chunks = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = self._text(item[0])
                body = item[1]
                if isinstance(body, list):
                    body = " ".join(self._text(v) for v in body)
                else:
                    body = self._text(body)
                chunks.append(f"{title}\n{body}".strip())
            else:
                chunks.append(self._text(item))

        return "\n\n".join(chunks)

    def load_anah(self):
        ds = load_dataset("opencompass/anah")

        rows = []
        for row in ds["train"]:
            rows.append({
                "id": self._text(row["name"]),
                "question": self._text(row["selected_questions"]),
                "answer": "",
                "reference": self._text(row["documents"]),
            })

        return DatasetDict({
            "train": Dataset.from_list(rows)
        })

    def load_faitheval_inconsistent(self):
        ds = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")

        rows = []
        for row in ds:
            answers = row["answers"]
            answer = ""
            if isinstance(answers, list) and len(answers) > 0:
                answer = self._text(answers[0])
            else:
                answer = self._text(answers)

            rows.append({
                "id": self._text(row["qid"]),
                "question": self._text(row["question"]),
                "answer": answer,
                "reference": self._first_faitheval_document(row["context"]),
            })

        return Dataset.from_list(rows)

    def load_truthfulqa(self):
        ds = load_dataset("domenicrosati/TruthfulQA")

        rows = []
        for row in ds["train"]:
            rows.append({
                "id": self._text(row.get("Question", "")),
                "question": self._text(row["Question"]),
                "answer": self._text(row["Best Answer"]),
                "reference": "",
            })

        return DatasetDict({
            "train": Dataset.from_list(rows)
        })

    def load_halubench(self):
        ds = load_dataset("PatronusAI/HaluBench")

        rows = []
        for row in ds["test"]:
            rows.append({
                "id": self._text(row.get("id", "")),
                "question": self._text(row["question"]),
                "answer": self._text(row["answer"]),
                "reference": self._text(row["passage"]),
            })

        return DatasetDict({
            "test": Dataset.from_list(rows)
        })

    def load_ragtruth(self):
        ds = load_dataset("wandb/RAGTruth-processed")

        rows = []
        for row in ds["train"]:
            rows.append({
                "id": self._text(row.get("id", "")),
                "question": self._text(row["query"]),
                "answer": self._text(row["output"]),
                "reference": self._text(row["context"]),
            })

        return DatasetDict({
            "train": Dataset.from_list(rows)
        })

    def load_placebobench(self):
        ds = load_dataset("blue-guardrails/PlaceboBench")

        out = {}
        for split in ds.keys():
            rows = []
            for row in ds[split]:
                rows.append({
                    "id": self._text(row.get("id", "")),
                    "question": self._text(row["question"]),
                    "answer": self._text(row["answer"]),
                    "reference": self._text(row["retrieved_context"]),
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)

    def load_halueval_qa(self):
        ds = load_dataset("pminervini/HaluEval", "qa")

        out = {}
        for split in ds.keys():
            rows = []
            for row in ds[split]:
                rows.append({
                    "id": self._text(row.get("id", "")),
                    "question": self._text(row["question"]),
                    "answer": self._text(row["right_answer"]),
                    "reference": self._text(row["knowledge"]),
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)

    def load_triviaqa_rc_wikipedia(self):
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")

        out = {}
        for split in ds.keys():
            rows = []
            for row in ds[split]:
                answer = ""
                if isinstance(row["answer"], dict):
                    answer = self._text(row["answer"].get("value", ""))
                else:
                    answer = self._text(row["answer"])

                reference = ""
                if isinstance(row["entity_pages"], dict):
                    reference = self._text(row["entity_pages"].get("wiki_context", ""))
                else:
                    reference = self._text(row["entity_pages"])

                rows.append({
                    "id": self._text(row.get("question_id", row.get("id", ""))),
                    "question": self._text(row["question"]),
                    "answer": answer,
                    "reference": reference,
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)

    def load_drop(self):
        ds = load_dataset("ucinlp/drop")

        out = {}
        for split in ["train", "validation"]:
            rows = []
            for row in ds[split]:
                rows.append({
                    "id": self._text(row["section_id"]),
                    "question": self._text(row["question"]),
                    "answer": self._text(row["answers_spans"]),
                    "reference": self._text(row["passage"]),
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)

    def load_hotpotqa_fullwiki(self):
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

        out = {}
        for split in ["train", "validation", "test"]:
            rows = []
            for row in ds[split]:
                rows.append({
                    "id": self._text(row["id"]),
                    "question": self._text(row["question"]),
                    "answer": self._text(row.get("answer", "")),
                    "reference": self._hotpot_reference(row["context"]),
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)

    def load_2wikimultihopqa(self):
        ds = load_dataset("framolfese/2WikiMultihopQA")

        out = {}
        for split in ["train", "validation", "test"]:
            rows = []
            for row in ds[split]:
                rows.append({
                    "id": self._text(row["id"]),
                    "question": self._text(row["question"]),
                    "answer": self._text(row["answer"]),
                    "reference": self._twowiki_reference(row["context"]),
                })
            out[split] = Dataset.from_list(rows)

        return DatasetDict(out)


if __name__ == "__main__":
    loader = HallucinationDatasetLoader()

    anah = loader.load_anah()
    print("\n=== ANAH ===")
    print(anah)
    print(anah["train"][0])

    faitheval = loader.load_faitheval_inconsistent()
    print("\n=== FaithEval inconsistent ===")
    print(faitheval)
    print(faitheval[0])

    truthfulqa = loader.load_truthfulqa()
    print("\n=== TruthfulQA ===")
    print(truthfulqa)
    print(truthfulqa["train"][0])

    halubench = loader.load_halubench()
    print("\n=== HaluBench ===")
    print(halubench)
    print(halubench["test"][0])

    ragtruth = loader.load_ragtruth()
    print("\n=== RAGTruth ===")
    print(ragtruth)
    print(ragtruth["train"][0])

    placebobench = loader.load_placebobench()
    print("\n=== PlaceboBench ===")
    print(placebobench)
    first_split = list(placebobench.keys())[0]
    print(placebobench[first_split][0])

    halueval = loader.load_halueval_qa()
    print("\n=== HaluEval QA ===")
    print(halueval)
    first_split = list(halueval.keys())[0]
    print(halueval[first_split][0])

    triviaqa = loader.load_triviaqa_rc_wikipedia()
    print("\n=== TriviaQA rc.wikipedia ===")
    print(triviaqa)
    first_split = list(triviaqa.keys())[0]
    print(triviaqa[first_split][0])

    drop_ds = loader.load_drop()
    print("\n=== DROP ===")
    print(drop_ds)
    print(drop_ds["train"][0])
    print(drop_ds["validation"][0])

    hotpotqa = loader.load_hotpotqa_fullwiki()
    print("\n=== HotpotQA fullwiki ===")
    print(hotpotqa)
    print(hotpotqa["train"][0])
    print(hotpotqa["validation"][0])
    print(hotpotqa["test"][0])

    twowiki = loader.load_2wikimultihopqa()
    print("\n=== 2WikiMultihopQA ===")
    print(twowiki)
    print(twowiki["train"][0])
    print(twowiki["validation"][0])
    print(twowiki["test"][0])