import re
from google.genai import Client


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
    r"pretend\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|guidelines)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
]

# Broad set of GitLab-related keywords for topic detection
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
]


def is_prompt_injection(query: str) -> bool:
    """Check if the query contains prompt injection patterns."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def is_on_topic(query: str, api_key: str = None) -> bool:
    """Check if the query is related to GitLab topics."""
    if not query or not query.strip():
        return False

    query_lower = query.lower()

    # Fast local keyword check
    if any(kw in query_lower for kw in GITLAB_TOPIC_KEYWORDS):
        return True

    # If no API key, rely on keyword check only
    if not api_key:
        return True  # Permissive default when no API key

    # Zero-shot classification using Gemini
    try:
        client = Client(api_key=api_key)
        prompt = (
            "Determine if the user query is related to GitLab company, "
            "its handbook, product, engineering, culture, hiring, remote work, "
            "values, or internal processes. "
            "Respond with only YES or NO.\n\n"
            f"Query: {query}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        answer = response.text.strip().upper()
        return "YES" in answer
    except Exception:
        # On API error, default to allowing the query
        return True


def verify_response_grounded(
    response_text: str, context_chunks: list, api_key: str
) -> bool:
    """Check if the response is grounded in the provided context chunks."""
    if not context_chunks:
        return True

    try:
        client = Client(api_key=api_key)
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
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        eval_res = response.text.strip().upper()
        print(f"[Guardrail] Fact-checker verdict: {eval_res}")
        # If the checker says UNSAFE, the response has fabricated content
        # Everything else (SAFE, "cannot determine", etc.) → safe
        return "UNSAFE" not in eval_res
    except Exception:
        # On API error, default to allowing the response
        return True
