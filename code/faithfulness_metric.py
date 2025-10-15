# utf-8
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# templates
class FaithfulnessTemplate:
    @staticmethod
    def generate_statements(actual_output: str) -> str:
        return f"""Given the text, breakdown and generate a list of statements presented. Ambiguous statements and single words can be considered as statements, but only if outside of a coherent statement.

Example:
Example text: 
Our new laptop model features a high-resolution Retina display for crystal-clear visuals. It also includes a fast-charging battery, giving you up to 12 hours of usage on a single charge. For security, we’ve added fingerprint authentication and an encrypted SSD. Plus, every purchase comes with a one-year warranty and 24/7 customer support.

{{
    "statements": [
        "The new laptop model has a high-resolution Retina display.",
        "It includes a fast-charging battery with up to 12 hours of usage.",
        "Security features include fingerprint authentication and an encrypted SSD.",
        "Every purchase comes with a one-year warranty.",
        "24/7 customer support is included."
    ]
}}
===== END OF EXAMPLE ======
        
**
IMPORTANT: Please make sure to only return in valid and parseable JSON format, with the "statements" key mapping to a list of strings. No words or explanation are needed. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.
**

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(context: str, statements_json: str) -> str:
        return f"""Your task is to judge the faithfulness of a series of statements based on the given context.
For EACH statement return 1 if the statement CAN be directly inferred from the context, otherwise return 0.
Return ONLY valid JSON with the following exact schema (no prose):

{{
  "verdicts": [ 0 or 1 for each statement in order ]
}}

Guidelines:
- "Directly inferred" means the statement is explicitly supported or is an unambiguous entailment of the context.
- Do NOT guess beyond the provided context.
- The number of verdicts MUST equal the number of statements.

Context:
<<<
{context}
>>>

Statements (JSON):
{statements_json}

JSON:
"""


# parsing aux funcs


def _safe_json_loads(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            candidate = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return {}

def _ensure_list_str(x) -> List[str]:
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


@dataclass
class FaithfulnessJudge:
    """
    1) statements = LLM(actual_output -> JSON["statements"])
    2) verdicts   = LLM(context + statements -> JSON["verdicts"] de 0/1)
    3) score      = (# de 1) / (nº de statements)
    """
    model_statements: str = "gpt-4.1-nano"
    model_verdicts: str = "gpt-4.1-nano"
    max_statements: int = 50

    def __post_init__(self):
        # Modelos nano no aceptan temperature/top_p => no los pasamos
        self.llm_statements = ChatOpenAI(model=self.model_statements)
        self.llm_verdicts = ChatOpenAI(model=self.model_verdicts)

    def _llm_json(self, llm: ChatOpenAI, prompt: str) -> Dict[str, Any]:
        msgs = [
            SystemMessage(content="You are a careful, JSON-only generator. Always return valid JSON."),
            HumanMessage(content=prompt),
        ]
        out = llm.invoke(msgs).content
        data = _safe_json_loads(out)
        return data if isinstance(data, dict) else {}

    def generate_statements(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.generate_statements(actual_output)
        data = self._llm_json(self.llm_statements, prompt)
        statements = _ensure_list_str(data.get("statements", []))
        return statements[: self.max_statements]

    def judge_verdicts(self, context_blocks: List[str], statements: List[str]) -> List[int]:
        context = " ".join([c for c in context_blocks if isinstance(c, str)]) if context_blocks else ""
        statements_json = json.dumps({"statements": statements}, ensure_ascii=False)
        prompt = FaithfulnessTemplate.generate_verdicts(context, statements_json)
        data = self._llm_json(self.llm_verdicts, prompt)

        raw = data.get("verdicts", [])
        # normaliza a lista de 0/1
        verdicts: List[int] = []
        for v in raw:
            try:
                verdicts.append(1 if int(v) == 1 else 0)
            except Exception:
                # si viene "yes"/"no"
                vs = str(v).strip().lower()
                verdicts.append(1 if vs in {"1", "yes", "true", "supported"} else 0)

        # ajusta longitud
        verdicts = verdicts[: len(statements)]
        while len(verdicts) < len(statements):
            verdicts.append(0)
        return verdicts

    def score(self, contexts: List[str], actual_output: str) -> Dict[str, Any]:
        statements = self.generate_statements(actual_output)
        if not statements:
            return {"score": 0.0, "n_statements": 0, "n_supported": 0, "statements": [], "verdicts": []}

        verdicts = self.judge_verdicts(contexts, statements)
        n_supported = sum(1 for v in verdicts if v == 1)
        n_total = len(statements)
        score = n_supported / n_total if n_total else 0.0

        return {
            "score": score,
            "n_statements": n_total,
            "n_supported": n_supported,
            "statements": statements,
            "verdicts": verdicts,  # lista de 0/1 por statement
        }


if __name__ == "__main__":
    import os

    judge = FaithfulnessJudge(
        model_statements="gpt-4.1-nano",
        model_verdicts="gpt-4.1-nano",
        max_statements=20,
    )

    actual_output = "White fish improves heart health due to omega-3. It contains protein and vitamins."
    contexts = [
        "Fish is a valuable source of high quality protein, minerals and vitamins.",
        "Oily fish are rich in omega-3 PUFA which can reduce coronary heart disease risk."
    ]

    res = judge.score(contexts, actual_output)
    print(json.dumps(res, indent=2, ensure_ascii=False))
