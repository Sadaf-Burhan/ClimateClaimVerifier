"""
Stage 2: Topic Filter

Lightweight keyword-based filter that runs BEFORE any LLM processing.
A post must pass both checks to proceed to the claim extraction stage:

  1. WEATHER/CLIMATE CHECK  — does the post mention a relevant climate or
                               extreme weather concept?
  2. NORTH AMERICA CHECK    — does the post reference NA geography, OR does
                               it contain no clear non-NA geographic signal?
                               (Posts with no geography are kept — they are
                               likely general climate discussions.)

This is intentionally cheap and fast. It is not perfect — borderline posts
will be caught or dropped by the LLM in stage 4 (claim extraction).
The goal here is to discard obvious noise: sports scores, celebrity gossip,
foreign-only events, and off-topic posts before burning LLM tokens on them.
"""

import re

# ── Weather / Climate Terms ───────────────────────────────────────────────────
# A post must contain at least one of these to be considered relevant.

WEATHER_TERMS = {
    # General climate
    "climate", "global warming", "greenhouse", "carbon", "emission",
    "net zero", "geoengineering", "tipping point", "climate change",
    # Extreme weather events
    "wildfire", "wildfire smoke", "forest fire", "bushfire",
    "flood", "flooding", "flash flood",
    "tornado", "funnel cloud", "twister",
    "hurricane", "typhoon", "tropical storm", "cyclone",
    "heat dome", "heat wave", "heatwave", "extreme heat",
    "blizzard", "snowstorm", "winter storm", "ice storm",
    "drought", "megadrought",
    "atmospheric river", "bomb cyclone", "derecho",
    "storm surge", "superstorm",
    # Ice / sea
    "sea level", "ice sheet", "glacier", "arctic", "permafrost",
    "greenland melt", "antarctica",
    # Misinformation targets
    "climate hoax", "weather manipulation", "chemtrail", "haarp",
    "climate lockdown", "cloud seeding",
    "methane bomb", "jet stream collapse",
}

# ── North America Geography ───────────────────────────────────────────────────
# If any of these appear the post is confirmed NA-relevant.
# If NONE of these appear AND no FOREIGN_TERMS appear → keep (assume general).
# If FOREIGN_TERMS appear and no NA_TERMS appear → discard.

NA_TERMS = {
    # Countries
    "canada", "united states", "usa", "u.s.", "u.s.a", "america", "north america",
    # US regions / states (common in weather discussions)
    "california", "texas", "florida", "louisiana", "alabama", "mississippi",
    "georgia", "south carolina", "north carolina", "virginia", "maryland",
    "new york", "new jersey", "connecticut", "massachusetts",
    "ohio", "michigan", "illinois", "indiana", "wisconsin", "minnesota",
    "iowa", "missouri", "kansas", "nebraska", "oklahoma",
    "montana", "wyoming", "colorado", "utah", "nevada", "arizona",
    "washington", "oregon", "idaho",
    "alaska", "hawaii",
    "midwest", "southeast", "gulf coast", "east coast", "west coast",
    "pacific northwest", "great plains", "tornado alley", "dixie alley",
    "appalachian", "rocky mountains", "sierra nevada",
    # Canadian provinces / regions
    "alberta", "british columbia", "ontario", "quebec", "manitoba",
    "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
    "northwest territories", "yukon", "nunavut",
    "bc wildfire", "prairie", "canadian rockies",
    # Major NA cities often mentioned in weather events
    "houston", "miami", "new orleans", "phoenix", "las vegas",
    "los angeles", "san francisco", "seattle", "portland",
    "chicago", "detroit", "toronto", "vancouver", "calgary",
    "edmonton", "winnipeg", "montreal",
}

# ── Foreign Geography Signals ─────────────────────────────────────────────────
# If a post contains ONLY these and no NA_TERMS, it is likely not NA-relevant.

FOREIGN_TERMS = {
    "china", "india", "pakistan", "bangladesh", "russia", "europe",
    "australia", "new zealand", "africa", "brazil", "amazon",
    "philippines", "indonesia", "vietnam", "taiwan", "japan", "korea",
    "middle east", "iran", "saudi arabia", "ukraine", "mediterranean",
    "siberia", "himalayas",
}


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for consistent matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def is_weather_relevant(text: str) -> bool:
    """Returns True if the post mentions at least one climate/weather term."""
    t = _normalise(text)
    return any(term in t for term in WEATHER_TERMS)


def is_na_relevant(text: str) -> bool:
    """
    Returns True if the post is plausibly about North America.
    Logic:
      - Contains an NA geographic term → True
      - Contains a foreign geographic term but no NA term → False
      - No geographic signal at all → True (keep for general climate discussion)
    """
    t = _normalise(text)
    has_na      = any(term in t for term in NA_TERMS)
    has_foreign = any(term in t for term in FOREIGN_TERMS)

    if has_na:
        return True
    if has_foreign:
        return False
    return True  # no geography mentioned — keep


def is_relevant(text: str) -> bool:
    """
    Master filter: returns True only if the post passes both checks.
    This is the single function called by the pipeline.
    """
    return is_weather_relevant(text) and is_na_relevant(text)


def filter_posts(posts: list[dict]) -> tuple[list[dict], int]:
    """
    Filter a list of post dicts.
    Returns (kept_posts, discarded_count).
    """
    kept      = [p for p in posts if is_relevant(p.get("text", ""))]
    discarded = len(posts) - len(kept)
    return kept, discarded
