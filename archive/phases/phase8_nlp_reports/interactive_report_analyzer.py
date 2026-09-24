from emergency_nlp import parse_emergency_report

def print_report(result):
    print("\n" + "=" * 50)
    print("EMERGENCY REPORT ANALYSIS")
    print("=" * 50)
    print(f"Original text: {result['original_text']}")
    print(f"Locations detected: {result['locations'] if result['locations'] else 'None found'}")
    print(f"People count: {result['people_count'] if result['people_count'] is not None else 'Not specified'}")
    print(f"Event type(s): {', '.join(result['event_types'])}")
    print(f"Severity: {result['severity']}")
    print("=" * 50 + "\n")


def main():
    print("AegisAI - Phase 8 Emergency Report Analyzer")
    print("Type an emergency message and press Enter to analyze it.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter emergency report: ").strip()

        if text.lower() in ("quit", "exit"):
            print("Exiting. Stay safe!")
            break

        if not text:
            print("Please enter some text.\n")
            continue

        result = parse_emergency_report(text)
        print_report(result)


if __name__ == "__main__":
    main()
