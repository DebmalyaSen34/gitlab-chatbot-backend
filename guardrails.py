import os
import re
from openai import OpenAI


INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?previous",
    r"ignore\s+(?:all\s+)?prior",
    r"system\s+prompt",
    r"reveal\s+(?:your\s+)?instructions",
    r"show\s+(?:me\s+)?(?:your\s+)?(?:system|initial)\s+(?:prompt|instructions)",
    r"you\s+are\s+now\s+a\s+system",
    r"bypass\s+(?:all\s+)?restrictions",
    r"override\s+(?:your\s+)?(?:safety|rules|guidelines)",
    r"disregard\s+(?:all\s+)?(?:your\s+)?(?:prior|previous|earlier)\s+(?:instructions|rules)",
    r"act\s+as\s+(?:if\s+)?you\s+(?:have\s+)?no\s+(?:restrictions|rules|limits)",
    r"pretend\s+you\s+are\s+(?:no|without)\s+(?:rules|restrictions|guidelines)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
]

GITLAB_TOPIC_KEYWORDS = [
    "gitlab", "handbook", "value", "values", "remote", "direction", "hiring",
    "onboarding", "ceo", "shadow", "process", "culture", "diversity",
    "inclusion", "transparency", "collaboration", "results", "efficiency",
    "iteration", "bias", "action", "code review", "merge request",
    "continuous integration", "ci/cd", "devops", "sre", "engineering",
    "product", "marketing", "sales", "finance", "legal", "security",
    "okr", "kpi", "dri", "directly responsible individual",
    "all-remote", "asynchronous", "async", "communication",
    "benefits", "compensation", "equity", "stock", "pto",
    "vacation", "parental leave", "learning", "development",
    "performance", "review", "feedback", "1:1", "one-on-one",
    "team", "department", "division", "group", "stage",
    "section", "infrastructure", "design", "ux", "research",
    "documentation", "wiki", "runbook", "playbook",
    "enterprise", "ai", "artificial intelligence", "machine learning",
    "strategy", "vision", "mission", "goal", "objective",
    "policy", "guideline", "principle", "standard",
    "customer", "user", "stakeholder", "partner",
    "release", "deploy", "feature", "bug", "issue",
    "sprint", "agile", "scrum", "kanban", "milestone",
    "salary", "wage", "bonus", "promotion", "career",
    "manager", "lead", "director", "vp", "executive",
    "meeting", "standup", "retrospective", "demo",
    "training", "mentor", "coach", "growth",
]

_llm_client: OpenAI | None = None


def _get_llm_client() -> OpenAI:
    """Get or create a shared OpenAI client."""
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        client_kwargs = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_API_BASE")
        if base_url:
            client_kwargs["base_url"] = base_url
        _llm_client = OpenAI(**client_kwargs)
    return _llm_client


def is_prompt_injection(query: str) -> bool:
    """Check if the query contains prompt injection patterns."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def is_on_topic(query: str) -> bool:
    """Topic filtering is handled by vector search similarity threshold."""
    return True


def verify_response_grounded(
    response_text: str, context_chunks: list) -> bool:
    """Check if the response is grounded in the provided context chunks."""
    if not context_chunks:
        return True

    try:
        client = _get_llm_client()
        llm_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo") #NOTE: Any other model can also be used
        context_str = "\n---\n".join(
            [c.get_content() if hasattr(c, "get_content") else str(c) for c in context_chunks]
        )
        prompt = (
            "You are evaluating whether a chatbot response is reasonably grounded "
            "in the provided context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Generated Answer:\n{response_text}\n\n"
            "Evaluate the Generated Answer:\n"
            "1. Do the answer's main claims align with the context? Minor paraphrasing, "
            "summarization, or reasonable inference from the context is acceptable.\n"
            "2. Does the answer introduce completely fabricated facts, specific numbers, "
            "or claims that are nowhere in the context?\n\n"
            "IMPORTANT: If the answer says it cannot find the information, cannot answer, "
            "or says the context does not contain enough information — that is always SAFE.\n\n"
            "Reply with exactly one word: SAFE or UNSAFE.\n"
            "SAFE = the answer is reasonably grounded (minor inference/paraphrasing is OK).\n"
            "UNSAFE = the answer introduces fabricated facts or significantly contradicts the context."
        )
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        eval_res = response.choices[0].message.content.strip().upper()
        print(f"[Guardrail] Fact-checker verdict: {eval_res}")
        return "UNSAFE" not in eval_res
    except Exception:
        return True
