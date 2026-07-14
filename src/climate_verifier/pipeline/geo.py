"""
Lightweight geolocation for region-aware evidence retrieval (Week 6 RAG improvement).

The Week 6 evidence retriever was a PLAIN dense query — topically similar news won,
regardless of region, so a British-Columbia wildfire claim pulled UK wildfire articles.
The fix constrains retrieval at the SOURCE with metadata instead of a post-hoc re-rank:
we derive a coarse "Region, Country" location for each GDELT article and embed it INTO
the vector document (and into the claim query), so dense retrieval prefers same-region news.

Two derivation signals, in priority order:
  1. place-names in the headline / claim text  (the EVENT location — most reliable;
     a UK outlet reporting on "British Columbia wildfires" is about BC, not the UK)
  2. the news domain                            (ccTLD + known-outlet hints — catches the
     case the user flagged: weatherbc.com whose *headline* names no place)

This is a SOFT signal (embedded text), never a hard `where` filter: a US outlet covering
a BC fire has no "bc" in its domain and its headline may name no place — a hard location
filter would silently drop that real coverage. Location nudges ranking; it never prunes.
"""

import re

# --- country-code TLD -> country (unambiguous ccTLDs only) -------------------
_TLD_COUNTRY = {
    "ca": "Canada", "uk": "United Kingdom", "au": "Australia", "nz": "New Zealand",
    "ie": "Ireland", "in": "India", "cn": "China", "jp": "Japan", "de": "Germany",
    "fr": "France", "es": "Spain", "it": "Italy", "za": "South Africa", "ru": "Russia",
    "br": "Brazil", "mx": "Mexico", "ph": "Philippines", "us": "United States",
}

# --- known outlet domains -> location (overrides TLD; catches .com/.org outlets) ---
# Keyed by a substring of the registrable domain. Order-independent (first hit wins).
_DOMAIN_LOC = {
    "weatherbc":   "British Columbia, Canada",
    "bbc":         "United Kingdom",
    "theguardian": "United Kingdom",
    "telegraph":   "United Kingdom",
    "independent": "United Kingdom",
    "dailymail":   "United Kingdom",
    "skynews":     "United Kingdom",
    "metoffice":   "United Kingdom",
    "cbc":         "Canada",
    "ctvnews":     "Canada",
    "globalnews":  "Canada",
    "weather.gc":  "Canada",
    "abc.net.au":  "Australia",
    "smh.com":     "Australia",
    "news.com.au": "Australia",
    "xinhua":      "China",
    "news.cn":     "China",
    "chinadaily":  "China",
}

# --- place-name -> canonical "Region, Country" (headline / claim text) --------
# Canadian provinces & territories, US states, and the big cross-region distinguishers.
_CA = "Canada"
_US = "United States"
_PLACES = {
    # Canadian provinces / territories
    "british columbia": f"British Columbia, {_CA}", "alberta": f"Alberta, {_CA}",
    "saskatchewan": f"Saskatchewan, {_CA}", "manitoba": f"Manitoba, {_CA}",
    "ontario": f"Ontario, {_CA}", "quebec": f"Quebec, {_CA}", "québec": f"Quebec, {_CA}",
    "nova scotia": f"Nova Scotia, {_CA}", "new brunswick": f"New Brunswick, {_CA}",
    "newfoundland": f"Newfoundland, {_CA}", "prince edward island": f"Prince Edward Island, {_CA}",
    "yukon": f"Yukon, {_CA}", "nunavut": f"Nunavut, {_CA}",
    "northwest territories": f"Northwest Territories, {_CA}",
    # Canadian cities
    "vancouver": f"British Columbia, {_CA}", "kelowna": f"British Columbia, {_CA}",
    "calgary": f"Alberta, {_CA}", "edmonton": f"Alberta, {_CA}", "fort mcmurray": f"Alberta, {_CA}",
    "toronto": f"Ontario, {_CA}", "ottawa": f"Ontario, {_CA}", "montreal": f"Quebec, {_CA}",
    "montréal": f"Quebec, {_CA}", "winnipeg": f"Manitoba, {_CA}", "halifax": f"Nova Scotia, {_CA}",
    # US states (full names — abbreviations handled separately, they false-positive)
    "california": f"California, {_US}", "texas": f"Texas, {_US}", "florida": f"Florida, {_US}",
    "arizona": f"Arizona, {_US}", "nevada": f"Nevada, {_US}", "oregon": f"Oregon, {_US}",
    "washington state": f"Washington, {_US}", "colorado": f"Colorado, {_US}",
    "new mexico": f"New Mexico, {_US}", "louisiana": f"Louisiana, {_US}",
    "oklahoma": f"Oklahoma, {_US}", "kansas": f"Kansas, {_US}", "nebraska": f"Nebraska, {_US}",
    "iowa": f"Iowa, {_US}", "missouri": f"Missouri, {_US}", "arkansas": f"Arkansas, {_US}",
    "tennessee": f"Tennessee, {_US}", "kentucky": f"Kentucky, {_US}", "alabama": f"Alabama, {_US}",
    "mississippi": f"Mississippi, {_US}", "georgia": f"Georgia, {_US}",
    "north carolina": f"North Carolina, {_US}", "south carolina": f"South Carolina, {_US}",
    "virginia": f"Virginia, {_US}", "louisiana": f"Louisiana, {_US}", "montana": f"Montana, {_US}",
    "idaho": f"Idaho, {_US}", "utah": f"Utah, {_US}", "wyoming": f"Wyoming, {_US}",
    "minnesota": f"Minnesota, {_US}", "wisconsin": f"Wisconsin, {_US}", "michigan": f"Michigan, {_US}",
    "illinois": f"Illinois, {_US}", "ohio": f"Ohio, {_US}", "pennsylvania": f"Pennsylvania, {_US}",
    "new york": f"New York, {_US}", "new jersey": f"New Jersey, {_US}",
    "massachusetts": f"Massachusetts, {_US}", "maine": f"Maine, {_US}", "alaska": f"Alaska, {_US}",
    "hawaii": f"Hawaii, {_US}", "north dakota": f"North Dakota, {_US}",
    "south dakota": f"South Dakota, {_US}",
    # US cities (common in weather headlines)
    "los angeles": f"California, {_US}", "san francisco": f"California, {_US}",
    "san diego": f"California, {_US}", "sacramento": f"California, {_US}",
    "houston": f"Texas, {_US}", "dallas": f"Texas, {_US}", "austin": f"Texas, {_US}",
    "phoenix": f"Arizona, {_US}", "miami": f"Florida, {_US}", "orlando": f"Florida, {_US}",
    "tampa": f"Florida, {_US}", "seattle": f"Washington, {_US}", "portland": f"Oregon, {_US}",
    "denver": f"Colorado, {_US}", "chicago": f"Illinois, {_US}", "new orleans": f"Louisiana, {_US}",
    # Cross-region distinguishers — countries / big regions. Foreign-event headlines MUST
    # resolve to the event country, else a domestic outlet covering them (cbc.ca on a Spain
    # wildfire) would wrongly fall back to the outlet's home country.
    "united states": _US, "usa": _US, "u.s.a": _US,
    "united kingdom": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom", "london": "United Kingdom",
    "australia": "Australia", "sydney": "Australia", "melbourne": "Australia",
    "new zealand": "New Zealand", "india": "India", "china": "China", "japan": "Japan",
    "spain": "Spain", "portugal": "Portugal", "greece": "Greece", "italy": "Italy",
    "france": "France", "germany": "Germany", "turkey": "Turkey", "turkiye": "Turkey",
    "morocco": "Morocco", "algeria": "Algeria", "tunisia": "Tunisia", "egypt": "Egypt",
    "ireland": "Ireland", "netherlands": "Netherlands", "belgium": "Belgium",
    "russia": "Russia", "ukraine": "Ukraine", "brazil": "Brazil", "argentina": "Argentina",
    "chile": "Chile", "mexico": "Mexico", "south africa": "South Africa",
    "indonesia": "Indonesia", "pakistan": "Pakistan", "bangladesh": "Bangladesh",
    "philippines": "Philippines", "vietnam": "Vietnam", "thailand": "Thailand",
    "europe": "Europe", "africa": "Africa",
}

# Strict abbreviations: uppercase, word-boundary, on the ORIGINAL-case text only.
# (Lower-case "bc"/"ca" appear inside ordinary words — these must not false-positive.)
_ABBR = {"BC": f"British Columbia, {_CA}", "B.C.": f"British Columbia, {_CA}",
         "US": _US, "U.S.": _US, "U.S.A.": _US}

# Longest phrase first so "washington state" beats "washington", "new york" is whole, etc.
_PLACE_ORDER = sorted(_PLACES, key=len, reverse=True)
_PLACE_RE = {name: re.compile(rf"\b{re.escape(name)}\b") for name in _PLACE_ORDER}
_ABBR_RE = {a: re.compile(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])") for a in _ABBR}


def _domain_location(domain: str) -> str:
    """Coarse location from a news domain: known-outlet hint, else ccTLD -> country."""
    d = re.sub(r"^https?://", "", (domain or "").lower().strip()).split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    if not d:
        return ""
    for key, loc in _DOMAIN_LOC.items():
        if key in d:
            return loc
    parts = d.split(".")
    # second-level ccTLDs (co.uk, com.au, gov.au, net.nz, ...)
    if len(parts) >= 2 and parts[-2] in ("co", "com", "net", "org", "gov", "ac") \
            and parts[-1] in _TLD_COUNTRY:
        return _TLD_COUNTRY[parts[-1]]
    if parts and parts[-1] in _TLD_COUNTRY:
        return _TLD_COUNTRY[parts[-1]]
    return ""


def _text_location(text: str) -> str:
    """Coarse location from place-names in headline / claim text (the EVENT location)."""
    if not text:
        return ""
    low = text.lower()
    for name in _PLACE_ORDER:
        if _PLACE_RE[name].search(low):
            return _PLACES[name]
    for abbr, loc in _ABBR.items():          # strict, original-case
        if _ABBR_RE[abbr].search(text):
            return loc
    return ""


def extract_location(text: str, domain: str = "") -> str:
    """Best-effort 'Region, Country' for a news article or a claim.

    Headline/claim place-names win over the domain (they name the EVENT location, not the
    outlet's home country). Falls back to the domain when the text names no place — the
    weatherbc.com case. Returns "" when nothing is found (then location is simply omitted)."""
    return _text_location(text) or _domain_location(domain)


def with_location(text: str, location: str) -> str:
    """The exact document/query string that gets embedded. Used IDENTICALLY at index time
    (article headline) and query time (claim text) so the location token lands in the same
    place in both vectors — that alignment is what makes same-region news rank higher."""
    text = (text or "").strip()
    return f"{text}\nLocation: {location}" if location else text
