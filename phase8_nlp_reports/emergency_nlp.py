import spacy
import re

nlp = spacy.load("en_core_web_sm")

EVENT_KEYWORDS = {
    "FIRE": ["fire", "smoke", "flame", "flames", "burning"],
    "FLOOD": ["flood", "flooding", "water rising", "submerged"],
    "MEDICAL": ["injured", "unconscious", "bleeding", "heart attack", "someone collapsed", "person collapsed"],
    "STRUCTURAL": ["building has collapsed", "building collapsed", "cracked", "structural damage", "debris"],
    "TRAPPED": ["trapped", "stuck", "can't get out", "blocked in"],
}

SEVERITY_KEYWORDS = {
    "CRITICAL": ["trapped", "unconscious", "dying", "critical", "severe", "collapsed"],
    "HIGH": ["injured", "bleeding", "smoke", "fire", "spreading"],
    "MODERATE": ["concerned", "worried", "smell", "minor"],
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def extract_locations(doc):
    locations = []
    for ent in doc.ents:
        if ent.label_ in ("GPE", "FAC", "LOC"):
            locations.append(ent.text)
    return locations


def extract_people_count(text):
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b\s+(people|persons|individuals|victims)",
        text, re.IGNORECASE
    )
    if not match:
        return None

    num_str = match.group(1).lower()
    if num_str.isdigit():
        return int(num_str)
    return NUMBER_WORDS.get(num_str)


def detect_event_type(text):
    text_lower = text.lower()
    detected = []
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(event_type)
    return detected if detected else ["UNKNOWN"]


def assess_severity(text):
    text_lower = text.lower()
    for level, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return "LOW"


def parse_emergency_report(text):
    doc = nlp(text)

    return {
        "original_text": text,
        "locations": extract_locations(doc),
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
    ]

    for i, report in enumerate(test_reports, 1):
        result = parse_emergency_report(report)
        print(f"=== Report {i} ===")
        print(f"Text: {result['original_text']}")
        print(f"Locations: {result['locations']}")
        print(f"People count: {result['people_count']}")
        print(f"Event type(s): {result['event_types']}")
        print(f"Severity: {result['severity']}\n")
