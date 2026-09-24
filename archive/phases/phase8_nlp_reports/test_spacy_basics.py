import spacy

nlp = spacy.load("en_core_web_sm")

test_message = "There is smoke coming from the second floor near the laboratory in Building 4. Three people are trapped near Gate 4."

doc = nlp(test_message)

print("=== spaCy's built-in entity recognition ===")
for ent in doc.ents:
    print(f"  {ent.text} -> {ent.label_}")

print("\n=== All tokens with part-of-speech tags ===")
for token in doc:
    if not token.is_stop and not token.is_punct:
        print(f"  {token.text} ({token.pos_})")
