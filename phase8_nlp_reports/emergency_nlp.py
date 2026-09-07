import spacy
import re

nlp = spacy.load("en_core_web_sm")

EVENT_KEYWORDS = {
    "FIRE": ["fire", "flame", "flames", "burning", "visible smoke", "smoke coming from", "smoke billowing"],
    "POSSIBLE_FIRE": ["smoke smell", "smell of smoke", "smells like smoke", "smoky smell"],
    "FLOOD": ["flood", "flooding", "water rising", "submerged"],
    "MEDICAL": ["injured", "unconscious", "bleeding", "heart attack", "someone collapsed", "person collapsed"],
    "STRUCTURAL": ["building has collapsed", "building collapsed", "cracked", "structural damage", "debris"],
    "TRAPPED": ["trapped", "stuck", "can't get out", "blocked in"],
}

SEVERITY_KEYWORDS = {
    "CRITICAL": ["trapped", "unconscious", "dying", "critical", "severe", "collapsed"],
    "HIGH": ["injured", "bleeding", "fire", "spreading", "visible smoke", "smoke coming from"],
    "MODERATE": ["concerned", "worried", "smell", "smoke smell"],
}

DOWNPLAY_PHRASES = [
    "not urgent", "not major", "nothing major", "minor", "small",
    "just want someone to check", "no rush", "not serious", "not a big deal"
]

# Ordered so longer/more specific words are checked first (e.g. "laboratory" before "lab")
COMMON_ROOM_WORDS = [
    "laboratory", "classroom", "break room", "storage room", "parking lot",
    "bathroom", "bedroom", "hallway", "corridor", "basement", "attic",
    "garage", "lobby", "cafeteria", "stairwell", "elevator", "rooftop",
    "roof", "warehouse", "kitchen", "office", "lab"
]

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def extract_locations(doc, text):
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "FAC", "LOC")]

    text_lower = text.lower()
    matched_spans = []

    for room in COMMON_ROOM_WORDS:
        idx = text_lower.find(room)
        if idx == -1:
            continue
        # skip if this match falls entirely inside an already-matched span
        # (e.g. skip "lab" if "laboratory" already matched at an overlapping position)
        overlap = any(start <= idx < end for start, end in matched_spans)
        if overlap:
            continue
        matched_spans.append((idx, idx + len(room)))
        if room not in [loc.lower() for loc in locations]:
            locations.append(room)

    return locations


def extract_people_count(text):
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b\s+(people|persons|individuals|victims)",
        text, re.IGNORECASE
    )
    if not match:
        return None
    num_str = match.group(1).lower()
    return int(num_str) if num_str.isdigit() else NUMBER_WORDS.get(num_str)


def detect_event_type(text):
    text_lower = text.lower()
    detected = []
    has_possible_fire = any(kw in text_lower for kw in EVENT_KEYWORDS["POSSIBLE_FIRE"])
    has_real_fire = any(kw in text_lower for kw in EVENT_KEYWORDS["FIRE"])

    if has_real_fire:
        detected.append("FIRE")
    elif has_possible_fire:
        detected.append("POSSIBLE_FIRE")

    for event_type in ["FLOOD", "MEDICAL", "STRUCTURAL", "TRAPPED"]:
        if any(kw in text_lower for kw in EVENT_KEYWORDS[event_type]):
            detected.append(event_type)

    return detected if detected else ["UNKNOWN"]


def assess_severity(text):
    text_lower = text.lower()
    is_downplayed = any(phrase in text_lower for phrase in DOWNPLAY_PHRASES)

    for level in ["CRITICAL", "HIGH", "MODERATE"]:
        if any(kw in text_lower for kw in SEVERITY_KEYWORDS[level]):
            if is_downplayed and level in ("CRITICAL", "HIGH"):
                return "MODERATE"
            return level

    return "LOW"


def parse_emergency_report(text):
    doc = nlp(text)
    return {
        "original_text": text,
        "locations": extract_locations(doc, text),
        "people_count": extract_people_count(text),
        "event_types": detect_event_type(text),
        "severity": assess_severity(text),
    }


if __name__ == "__main__":
    test_reports = [
        "There is smoke coming from the second floor near the laboratory in Building 4. Three people are trapped near Gate 4.",
        "I see a small fire in the kitchen, nothing major, just want someone to check it out.",
        "The building has collapsed and there are people trapped under the debris near Main Street.",
        "Someone collapsed and is unconscious near the cafeteria entrance.",
        "Minor smoke smell in the break room, not urgent.",
    ]

    for i, report in enumerate(test_reports, 1):
        result = parse_emergency_report(report)
        print(f"=== Report {i} ===")
        print(f"Text: {result['original_text']}")
        print(f"Locations: {result['locations']}")
        print(f"People count: {result['people_count']}")
        print(f"Event type(s): {result['event_types']}")
        print(f"Severity: {result['severity']}\n")
