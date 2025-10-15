from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ContextSufficiencyTemplate:
    @staticmethod
    def generate_requirements(question: str) -> str:
        return f"""Break down the question into the MINIMAL list of factual requirements that must be present in the context to answer it correctly.
Return ONLY valid JSON with a single key "requirements" mapping to a list of short strings (atomic facts). Keep each requirement concise.

Example:
Question:
"What are the main dietary sources of vitamin C?"

{{
  "requirements": [
    "Foods that are high in vitamin C",
    "Examples of such foods"
  ]
}}

IMPORTANT:
- Return ONLY JSON.
- Keep the list minimal but sufficient.
- No explanations.

Question:
{question}

JSON:
"""

    @staticmethod
    def check_coverage(context_text: str, requirements_json: str) -> str:
        return f"""You are given a CONTEXT and a list of REQUIREMENTS needed to answer a question.
For EACH requirement, return 1 if the requirement CAN be directly supported or inferred from the CONTEXT, otherwise 0.
Return ONLY valid JSON with the following schema:

{{
  "coverage": [0 or 1 for each requirement in order]
}}

Guidelines:
- "Supported" means the requirement is explicitly present or is an unambiguous entailment of the CONTEXT.
- Do NOT assume facts not present in the CONTEXT.
- The number of coverage items MUST equal the number of requirements.

CONTEXT:
<<<
{context_text}
>>>

REQUIREMENTS (JSON):
{requirements_json}

JSON:
"""

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
class ContextSufficiencyJudge:
    """
    1) requirements = LLM(question -> JSON["requirements"])
    2) coverage     = LLM(context + requirements -> JSON["coverage"] de 0/1)
    3) score        = (# de requisitos cubiertos) / (nº de requisitos)
    """
    model_requirements: str = "gpt-4.1-nano"
    model_coverage: str = "gpt-4.1-nano"
    max_requirements: int = 20

    def __post_init__(self):
        self.llm_reqs = ChatOpenAI(model=self.model_requirements)
        self.llm_cov  = ChatOpenAI(model=self.model_coverage)

    def _llm_json(self, llm: ChatOpenAI, prompt: str) -> Dict[str, Any]:
        msgs = [
            SystemMessage(content="You are a careful, JSON-only generator. Always return valid JSON."),
            HumanMessage(content=prompt),
        ]
        out = llm.invoke(msgs).content
        data = _safe_json_loads(out)
        return data if isinstance(data, dict) else {}

    def generate_requirements(self, question: str) -> List[str]:
        prompt = ContextSufficiencyTemplate.generate_requirements(question)
        data = self._llm_json(self.llm_reqs, prompt)
        reqs = _ensure_list_str(data.get("requirements", []))
        return reqs[: self.max_requirements]

    def judge_coverage(self, contexts: List[str], requirements: List[str]) -> List[int]:
        context_text = " ".join([c for c in contexts if isinstance(c, str)]) if contexts else ""
        reqs_json = json.dumps({"requirements": requirements}, ensure_ascii=False)
        prompt = ContextSufficiencyTemplate.check_coverage(context_text, reqs_json)
        data = self._llm_json(self.llm_cov, prompt)

        raw = data.get("coverage", [])
        coverage: List[int] = []
        for v in raw:
            try:
                coverage.append(1 if int(v) == 1 else 0)
            except Exception:
                vs = str(v).strip().lower()
                coverage.append(1 if vs in {"1", "yes", "true", "supported"} else 0)

        coverage = coverage[: len(requirements)]
        while len(coverage) < len(requirements):
            coverage.append(0)
        return coverage

    def score(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        requirements = self.generate_requirements(question)
        if not requirements:
            return {"score": 0.0, "n_requirements": 0, "n_covered": 0,
                    "requirements": [], "coverage": []}

        coverage = self.judge_coverage(contexts, requirements)
        n_covered = sum(coverage)
        n_total = len(requirements)
        score = n_covered / n_total if n_total else 0.0
        missing = [req for req, cov in zip(requirements, coverage) if cov == 0]

        return {
            "score": score,
            "n_requirements": n_total,
            "n_covered": n_covered,
            "requirements": requirements,
            "coverage": coverage,      
            "missing_requirements": missing,
        }

if __name__ == "__main__":
    import os

    judge = ContextSufficiencyJudge(
        model_requirements="gpt-4.1-nano",
        model_coverage="gpt-4.1-nano",
        max_requirements=10,
    )

    question = "What are the main dietary sources of calcium and how much do adults need per day?"
    contexts = [
        "Calcium is a major mineral. We can find calcium in many animal and plant foods.",
        "Adults generally need around 1000 mg/day; older adults may need more.",
        "Examples include milk, yogurt, cheese, and some leafy greens."
    ]

    res = judge.score(question, contexts)
    print(json.dumps(res, indent=2, ensure_ascii=False))
