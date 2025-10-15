from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class AnswerRelevancyTemplate:
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
    def generate_verdicts(user_input: str, statements_json: str) -> str:
        return f"""For the provided list of statements, determine whether each statement is relevant to address the input.
Generate JSON objects with 'verdict' and 'reason' fields.
The 'verdict' should be 'yes' (relevant), 'no' (irrelevant), or 'idk' (ambiguous/supporting information).
Provide 'reason' ONLY for 'no' or 'idk' verdicts.
The statements are from an AI's actual output.

**
IMPORTANT: Please make sure to only return in valid and parseable JSON format, with the 'verdicts' key mapping to a list of JSON objects. Ensure all strings are closed appropriately. Repair any invalid JSON before you output it.

Expected JSON format:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "explanation for irrelevance"
        }},
        {{
            "verdict": "idk",
            "reason": "explanation for ambiguity"
        }}
    ]  
}}

Generate ONE verdict per statement - number of 'verdicts' MUST equal number of statements.
'verdict' must be STRICTLY 'yes', 'no', or 'idk':
- 'yes': statement is relevant to addressing the input
- 'no': statement is irrelevant to the input  
- 'idk': statement is ambiguous (not directly relevant but could be supporting information)
Provide 'reason' ONLY for 'no' or 'idk' verdicts.
**          

Input:
{user_input}

Statements:
{statements_json}

JSON:
"""

def _safe_json_loads(s: str) -> Any:
    """
    Carga JSON robustamente. Si falla, intenta:
    - extraer el primer bloque {...} con llaves balanceadas
    - arreglar comas finales típicas
    """
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            candidate = match.group(0)

            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
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
class Verdict:
    verdict: str
    reason: Optional[str] = None

def _calculate_score(verdicts: List[Verdict], strict_mode: bool, threshold: float) -> float:
    """
    Tu lógica: cuenta como 'relevante' todo lo que NO sea 'no' (i.e., yes o idk).
    """
    n = len(verdicts)
    if n == 0:
        return 1.0
    relevant = sum(1 for v in verdicts if v.verdict.strip().lower() != "no")
    score = relevant / n
    return 0.0 if (strict_mode and score < threshold) else score

class AnswerRelevancyJudge:
    """
    Orquesta:
      1) statements = LLM(actual_output -> JSON["statements"])
      2) verdicts   = LLM(input + statements -> JSON["verdicts"])
      3) score      = proporción yes/idk sobre total
    """
    def __init__(
        self,
        model_statements: str = "gpt-4o-nano",
        model_verdicts: str = "gpt-4o-nano",
        strict_mode: bool = False,
        threshold: float = 0.5,
    ):
        self.llm_statements = ChatOpenAI(model=model_statements)
        self.llm_verdicts = ChatOpenAI(model=model_verdicts)
        self.strict_mode = strict_mode
        self.threshold = threshold

    def _llm_json(self, llm: ChatOpenAI, prompt: str) -> Dict[str, Any]:
        msgs = [
            SystemMessage(content="You are a careful, JSON-only generator. Always return valid JSON."),
            HumanMessage(content=prompt),
        ]
        out = llm.invoke(msgs).content
        data = _safe_json_loads(out)
        if not isinstance(data, dict):
            data = {}
        return data

    def generate_statements(self, actual_output: str, max_statements: int = 50) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(actual_output)
        data = self._llm_json(self.llm_statements, prompt)
        statements = _ensure_list_str(data.get("statements", []))
        # limitar por seguridad
        return statements[:max_statements]

    def judge_verdicts(self, user_input: str, statements: List[str]) -> List[Verdict]:
        statements_json = json.dumps({"statements": statements}, ensure_ascii=False)
        prompt = AnswerRelevancyTemplate.generate_verdicts(user_input, statements_json)
        data = self._llm_json(self.llm_verdicts, prompt)

        raw_verd = data.get("verdicts", [])
        verdicts: List[Verdict] = []
        for v in raw_verd:
            if isinstance(v, dict) and "verdict" in v:
                verdict = str(v["verdict"]).strip().lower()
                reason = v.get("reason", None)
                if verdict not in {"yes", "no", "idk"}:
                    verdict = "idk"  # normaliza valores raros
                verdicts.append(Verdict(verdict=verdict, reason=reason))
        verdicts = verdicts[: len(statements)]
        while len(verdicts) < len(statements):
            verdicts.append(Verdict(verdict="idk", reason="filled"))
        return verdicts

    def score(self, user_input: str, actual_output: str) -> Dict[str, Any]:
        statements = self.generate_statements(actual_output)
        verdicts = self.judge_verdicts(user_input, statements)
        score = _calculate_score(verdicts, self.strict_mode, self.threshold)
        return {
            "score": score,
            "n_statements": len(statements),
            "n_relevant": sum(1 for v in verdicts if v.verdict != "no"),
            "statements": statements,
            "verdicts": [v.__dict__ for v in verdicts],
        }

if __name__ == "__main__":
    import os

    judge = AnswerRelevancyJudge(
        model_statements="gpt-4.1-nano",
        model_verdicts="gpt-4.1-nano",
        strict_mode=False,
        threshold=0.5,
    )

    question = "What are the main dietary sources of vitamin C?"
    answer = "Oranges and strawberries are rich in vitamin C. Also, the Eiffel Tower is in Paris."

    res = judge.score(question, answer)
    print(json.dumps(res, indent=2, ensure_ascii=False))
