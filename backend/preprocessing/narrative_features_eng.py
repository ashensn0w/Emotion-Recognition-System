import spacy

nlp = spacy.load("en_core_web_sm")

def extract_mode(text):
    doc = nlp(text)
    has_mode = False  # Initialize a flag to check for mode features

    for token in doc:
        # Modal verbs and auxiliary verbs
        if token.pos_ == "AUX" or (token.pos_ == "VERB" and token.lemma_ in ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]):
            has_mode = True

        # Adverbs of necessity or possibility
        elif token.pos_ == "ADV" and token.lemma_ in ["necessarily", "probably", "possibly", "perhaps", "certainly"]:
            has_mode = True

        # Imperative mood: Root verb at the beginning of the sentence
        elif token.pos_ == "VERB" and token.dep_ == "ROOT" and token.i == 0:  # Checking if it's the first word
            has_mode = True

    # Return 1 if any mode feature was detected, else return 0
    return 1 if has_mode else 0

def extract_intention(text):
    doc = nlp(text)
    has_intention = False  # Initialize a flag to check for intention features

    for token in doc:
        # Verbs of intention or desire
        if token.pos_ == "VERB" and token.lemma_ in ["want", "need", "desire", "intend", "wish", "plan", "aim", "decide", "hope"]:
            has_intention = True

        # Infinitive phrases ('to' + verb)
        elif token.text == "to" and token.head.pos_ == "VERB" and token.dep_ == "aux":
            has_intention = True

        # Subordinate clauses of purpose (e.g., 'want to go', 'so that', 'in order to', 'so as to')
        elif token.text in ["so", "that", "order", "as"] and token.dep_ in ["mark", "advmod"]:
            # Check the head of the clause for verbs that indicate purpose
            if token.head.dep_ == "advcl" and token.head.head.pos_ == "VERB":
                has_intention = True

        # Auxiliary verbs indicating future actions
        elif token.pos_ == "AUX" and token.lemma_ in ["will", "shall"]:
            has_intention = True

    # Return 1 if any intention feature was detected, else return 0
    return 1 if has_intention else 0
# -----------------------------------------------------------------------------------------------------
# does not capture the "painted red" yet when the sentence "She painted the wall red." is used as an example NOT FINISHED, TRY NER
def extract_result(text):
    doc = nlp(text)
    result_features = []

    for token in doc:
        # Perfect tenses (have/has/had + past participle)
        if token.lemma_ in ["have", "has", "had"] and token.dep_ == "aux" and token.head.tag_ == "VBN":
            result_features.append(f"Perfect Tense: {token.head.text}")

        # Resultative constructions (verb + resultative complement - adjective or particle)
        elif token.pos_ == "VERB":
            for child in token.children:
                # Check for adjective or particle indicating a result (e.g., "wiped clean", "painted red")
                if child.pos_ in ["ADJ", "PART"]:
                    result_features.append(f"Resultative: {token.text} {child.text}")

        # Conjunctions of result (e.g., "so", "therefore")
        elif token.pos_ == "CCONJ" and token.lemma_ in ["so", "therefore"]:
            result_features.append(f"Conjunction: {token.text}")

        # Causal and sequential structures (e.g., "because", "since", "as", multi-word expression "as a result")
        elif token.pos_ == "SCONJ" and token.lemma_ in ["because", "since", "as"]:
            result_features.append(f"Structure: {token.text}")
        
        # Special case for multi-word "as a result"
        if "as a result" in text:
            result_features.append("Structure: as a result")

    return result_features
# -----------------------------------------------------------------------------------------------------
def extract_manner(text):
    doc = nlp(text)
    has_manner = False

    # Check for adverbs of manner
    for token in doc:
        if token.pos_ == "ADV" and token.dep_ == "advmod":
            has_manner = True
            break  # Exit early if we find at least one

    # Check for prepositional phrases of manner
    if has_manner == 0:  # Only continue checking if not already found
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ in ["pobj", "obj"]:
                        has_manner = True
                        break  # Exit early if we find at least one
                if has_manner:
                    break  # Exit early if we find at least one

    # Check for intensifiers or modifiers
    if has_manner == 0:  # Only continue checking if not already found
        for token in doc:
            if token.pos_ == "ADV" and token.dep_ == "advmod":
                if token.head.pos_ == "ADV":
                    has_manner = True
                    break  # Exit early if we find at least one

    return 1 if has_manner else 0

def extract_aspect(text):
    doc = nlp(text)
    has_aspect = False

    for token in doc:
        # Aspectual verbs (e.g., "start", "finish", "continue")
        if token.pos_ == "VERB" and token.lemma_ in ["start", "finish", "continue", "begin", "stop"]:
            has_aspect = True

        # Perfect tenses (have/has/had + past participle)
        if token.lemma_ in ["have", "has", "had"] and token.dep_ == "aux" and token.head.tag_ == "VBN":
            has_aspect = True

        # Progressive tenses (be + present participle)
        if token.lemma_ in ["be"] and token.dep_ == "aux" and token.head.tag_.startswith("VBG"):
            has_aspect = True

        # Aspectual adverbs or particles (e.g., "already", "yet", "still")
        if token.pos_ == "ADV" and token.lemma_ in ["already", "yet", "still"]:
            has_aspect = True

        # Auxiliary verbs indicating aspect (e.g., "will", "have", "be")
        if token.pos_ == "AUX" and token.lemma_ in ["will", "have", "be"]:
            has_aspect = True

    return 1 if has_aspect else 0

def extract_status(text):
    doc = nlp(text)
    has_status = False
    
    # Convert the document to a list of tokens
    tokens = [token.text for token in doc]

    # Define common multi-word negation phrases
    multi_word_negations = [
        "no longer", "not at all", "never again", "not really", "not yet", "not sure", "don't know"
    ]

    # Check for multi-word negation phrases
    for phrase in multi_word_negations:
        if phrase in text:
            has_status = True
            break  # No need to check further if a phrase is found

    # Check for other negation features
    if not has_status:  # Only check if not already found
        for token in doc:
            # Negation words (e.g., "not", "never", "no")
            if token.lemma_ in ["not", "never", "no"]:
                has_status = True
                break  # No need to check further if a negation word is found

            # Negation phrases involving auxiliary or verb
            if token.dep_ == "neg" and token.head.pos_ in ["AUX", "VERB"]:
                has_status = True
                break  # No need to check further if a negation phrase is found

            # Dependency relations involving negation
            if token.dep_ == "neg":
                has_status = True
                break  # No need to check further if a negation dependency is found

    return 1 if has_status else 0

def extract_appearance(text):
    doc = nlp(text)
    has_appearance = False

    # Check for conjunctions and linking words
    for token in doc:
        if token.pos_ == "CCONJ" or (token.pos_ == "PRON" and token.lemma_ in ["which", "that"]):
            has_appearance = True
            break  # Stop checking once the feature is found

    # Check for transformational verbs or phrases
    if not has_appearance:
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in ["become", "turn", "transform", "change"]:
                has_appearance = True
                break  # Stop checking once the feature is found

    # Check for negated transformational verbs
    if not has_appearance:
        for token in doc:
            if token.dep_ == "neg" and token.head.pos_ == "VERB" and token.head.lemma_ in ["become", "turn", "transform", "change"]:
                has_appearance = True
                break  # Stop checking once the feature is found

    return 1 if has_appearance else 0

def extract_knowledge(text):
    doc = nlp(text)
    has_knowledge = False  # Initialize as False

    # Define verbs related to knowledge
    knowledge_verbs = {"know", "realize", "remember", "learn", "recognize", "understand", "believe", "think", "see", "hear", "feel", "notice", "say", "tell", "inform", "report", "observe"}
    
    for token in doc:
        # Check if token is a verb related to knowledge
        if token.pos_ == "VERB" and token.lemma_ in knowledge_verbs:
            has_knowledge = True
            break  # No need to check further once a knowledge verb is found
        
        # Capture clauses starting with conjunctions related to knowledge
        if token.dep_ == "mark" and token.text.lower() in ["that"]:
            has_knowledge = True
            break  # No need to check further once a knowledge clause is found

        # Capture direct objects of knowledge verbs
        if token.dep_ == "dobj" and token.head.pos_ == "VERB" and token.head.lemma_ in knowledge_verbs:
            has_knowledge = True
            break  # No need to check further once a knowledge direct object is found

    return 1 if has_knowledge else 0

# -----------------------------------------------------------------------------------------------------
# # Example usage
# text = "I have already finished my homework."

# mode_features = extract_mode(text)
# intention_features = extract_intention(text)
# result_features = extract_result(text)
# manner_features = extract_manner(text)
# aspect_features = extract_aspect(text)
# status_features = extract_status(text)
# appearance_features = extract_appearance(text)
# knowledge_features = extract_knowledge(text)

# print(mode_features, intention_features, result_features, manner_features, aspect_features, status_features, appearance_features,
# knowledge_features)
# -----------------------------------------------------------------------------------------------------
# # MODE
# text = "You must finish your homework before watching TV."
# mode_features = extract_mode(text)
# print(mode_features)

# # INTENTION
# text = "I want to go to the beach tomorrow."
# intention_features = extract_intention(text)
# print(intention_features)

# # RESULT
# text = "He had finished his work before the deadline."
# result_features = extract_result(text)
# print(result_features)

# # MANNER
# text = "He ran very quickly."
# manner_features = extract_manner(text)
# print(manner_features)

# # ASPECT
# text = "I have already finished my homework."
# aspect_features = extract_aspect(text)
# print(aspect_features)

# # STATUS
# text = "He did not go to the party."
# status_features = extract_status(text)
# print(status_features)

# # APPEARANCE
# text = "He wore a suit which made him look professional."
# appearance_features = extract_appearance(text)
# print(appearance_features)

# # KNOWLEDGE
# text = "She knows that the project is complete"
# knowledge_features = extract_knowledge(text)
# print(knowledge_features)