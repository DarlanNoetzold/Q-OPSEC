import regex as re
from typing import List, Dict

class PatternsRepository:
    def __init__(self):
        # Simplified patterns
        self._patterns = {
            "credit_card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)"),
            "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "national_id": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"),
            "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
        }

    def scan(self, text: str) -> List[Dict[str, object]]:
        findings = []
        for name, pat in self._patterns.items():
            matches = [m.group(0) for m in pat.finditer(text or "")]
            if matches:
                findings.append({"type": name, "count": len(matches)})
        return findings

    def tags_from_findings(self, findings: List[Dict[str, object]]) -> List[str]:
        tags = set()
        for f in findings:
            if f["type"] in ("credit_card", "iban"):
                tags.add("finance")
                tags.add("PII")
            elif f["type"] in ("email", "national_id"):
                tags.add("PII")
        return sorted(list(tags))