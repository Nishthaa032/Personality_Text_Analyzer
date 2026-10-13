from __future__ import annotations

TRAIT_LABELS = {
    "agreeableness": "Agreeableness",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
}


def _level(score: float) -> str:
    if score >= 65:
        return "high"
    if score <= 35:
        return "low"
    return "moderate"


def build_personality_summary(traits: dict[str, float]) -> str:
    """Turn predicted scores into a cautious, plain-language interpretation."""
    levels = {trait: _level(score) for trait, score in traits.items()}
    highest = max(traits, key=traits.get)
    lowest = min(traits, key=traits.get)

    descriptions = {
        "agreeableness": {
            "high": "cooperative, considerate, and supportive",
            "moderate": "able to balance cooperation with independent opinions",
            "low": "direct, self-reliant, and comfortable expressing disagreement",
        },
        "openness": {
            "high": "curious, imaginative, and open to new ideas",
            "moderate": "comfortable with familiar approaches while remaining open to change",
            "low": "practical, consistent, and grounded in familiar approaches",
        },
        "conscientiousness": {
            "high": "organized, deliberate, and goal-focused",
            "moderate": "flexible while still able to plan and follow through",
            "low": "spontaneous and adaptable rather than strongly routine-driven",
        },
        "extraversion": {
            "high": "socially energetic and expressive",
            "moderate": "comfortable with both social time and time alone",
            "low": "reflective and more comfortable with quieter or smaller-group settings",
        },
        "neuroticism": {
            "high": "more emotionally sensitive to stress and uncertainty",
            "moderate": "likely to experience a typical range of emotional reactions",
            "low": "relatively calm and less reactive to everyday stress",
        },
    }

    profile = ", ".join(
        descriptions[trait][levels[trait]]
        for trait in TRAIT_LABELS
    )
    return (
        f"This text suggests a personality profile that is {profile}. "
        f"The strongest predicted dimension is {TRAIT_LABELS[highest]} "
        f"({traits[highest]:.0f}/100), while the lowest is {TRAIT_LABELS[lowest]} "
        f"({traits[lowest]:.0f}/100). These results are model-based indicators "
        "from the submitted text, not a clinical or definitive personality assessment."
    )
