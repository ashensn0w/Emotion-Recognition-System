import spacy

nlp = spacy.load("en_core_web_sm")

def extract_mode(text):
    doc = nlp(text)
    mode_features = []

    for token in doc:
        # Modal verbs
        if token.pos_ == "VERB" and token.lemma_ in ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]:
            mode_features.append(token.text)

        # Auxiliary verbs (e.g., 'do', 'have', 'be')
        elif token.pos_ == "AUX":
            mode_features.append(token.text)

        # Adverbs of necessity or possibility
        elif token.pos_ == "ADV" and token.lemma_ in ["necessarily", "probably", "possibly", "perhaps"]:
            mode_features.append(token.text)

        # Imperative mood (verb at the beginning of a sentence) but exclude other non-modal root verbs
        elif token.pos_ == "VERB" and token.dep_ == "ROOT" and token.lemma_ in ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]:
            mode_features.append(token.text)

    return mode_features

def extract_intention(text):
    doc = nlp(text)
    intention_features = []

    for token in doc:
        # Verbs of intention or desire
        if token.pos_ == "VERB" and token.lemma_ in ["want", "need", "desire", "intend", "wish"]:
            intention_features.append(token.text)

        # Infinitive phrases ('to' + verb)
        elif token.text == "to" and token.head.pos_ == "VERB":
            intention_features.append(f"Infinitive: {token.head.text}")

        # Subordinate clauses of purpose ('to' indicating purpose) tied to the head verb (e.g., 'want')
        elif token.text == "to" and token.dep_ == "aux" and token.head.pos_ == "VERB":
            intention_features.append(f"Purpose: {token.head.head.text}")

        # Auxiliary verbs indicating future actions
        elif token.pos_ == "AUX" and token.lemma_ in ["will", "shall"]:
            intention_features.append(token.text)

    return intention_features

def extract_result(text):
    doc = nlp(text)
    result_features = []

    for token in doc:
        # Perfect tenses (have/has/had + past participle)
        if token.lemma_ in ["have", "has", "had"] and token.dep_ == "aux" and token.head.tag_ == "VBN":
            result_features.append(f"Perfect Tense: {token.head.text}")

        # Resultative constructions (verb + resultative complement)
        elif token.pos_ == "VERB" and token.dep_ == "xcomp" and token.head.pos_ == "VERB":
            result_features.append(f"Resultative: {token.head.text} {token.text}")

        # Conjunctions of result (e.g., "so", "therefore")
        elif token.pos_ == "CCONJ" and token.lemma_ in ["so", "therefore"]:
            result_features.append(f"Conjunction: {token.text}")

        # Causal and sequential structures (e.g., "because", "since", "as a result")
        elif token.pos_ == "SCONJ" and token.lemma_ in ["because", "since", "as", "as a result"]:
            result_features.append(f"Structure: {token.text}")

    return result_features

def extract_manner(text):
    doc = nlp(text)
    manner_features = []

    for token in doc:
        # Adverbs of manner
        if token.pos_ == "ADV" and token.dep_ == "advmod":
            manner_features.append(token.text)

        # Prepositional phrases of manner (e.g., "in a hurry", "with care")
        elif token.pos_ == "ADP" and token.dep_ == "prep":
            manner_features.append(f"Prepositional Phrase: {token.text} {token.children[0].text}")

        # Intensifiers or modifiers (e.g., "very", "extremely")
        elif token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "ADV":
            manner_features.append(f"Modifier: {token.text} {token.head.text}")

    return manner_features

def extract_aspect(text):
    doc = nlp(text)
    aspect_features = []

    for token in doc:
        # Aspectual verbs (e.g., "start", "finish", "continue")
        if token.pos_ == "VERB" and token.lemma_ in ["start", "finish", "continue", "begin", "stop"]:
            aspect_features.append(token.text)

        # Perfect tenses (have/has/had + past participle)
        elif token.lemma_ in ["have", "has", "had"] and token.tag_.startswith("VBN"):
            aspect_features.append(f"Perfect Tense: {token.head.text}")

        # Progressive tenses (be + present participle)
        elif token.lemma_ in ["be"] and token.tag_.startswith("VBG"):
            aspect_features.append(f"Progressive Tense: {token.head.text}")

        # Aspectual adverbs or particles (e.g., "already", "yet", "still")
        elif token.pos_ == "ADV" and token.lemma_ in ["already", "yet", "still"]:
            aspect_features.append(token.text)

        # Auxiliary verbs indicating aspect (e.g., "will", "have", "be")
        elif token.pos_ == "AUX" and token.lemma_ in ["will", "have", "be"]:
            aspect_features.append(token.text)

    return aspect_features

def extract_status(text):
    doc = nlp(text)
    status_features = []

    for token in doc:
        # Negation words (e.g., "not", "never", "no")
        if token.lemma_ in ["not", "never", "no"]:
            status_features.append(token.text)

        # Negation phrases (e.g., "do not", "can't", "won't")
        elif token.lemma_ in ["do not", "cannot", "would not", "don't", "can't", "won't"]:
            status_features.append(token.text)

        # Dependency relations involving negation (e.g., "neg")
        elif token.dep_ == "neg":
            status_features.append(token.text)

        # Auxiliary verbs with negation
        elif token.pos_ == "AUX" and token.lemma_ in ["not", "never", "no"]:
            status_features.append(token.text)

    return status_features

# -----------------------------------------------------------------------------------------------------

def extract_appearance(text):
    doc = nlp(text)
    appearance_features = []

    for token in doc:
        # Conjunctions and linking words (e.g., "and", "but", "instead", "however", "which")
        if token.pos_ == "CCONJ" or token.pos_ == "PRON" and token.lemma_ in ["which", "that"]:
            appearance_features.append(token.text)

        # Transformational verbs or phrases (e.g., "become", "turn into")
        if token.pos_ == "VERB" and token.lemma_ in ["become", "turn"]:
            appearance_features.append(token.text)

        # Dependency relations indicating change with negation (make sure to use correct lemmas)
        if token.dep_ == "neg" and token.head.pos_ == "VERB" and token.head.lemma_ in ["become", "turn"]:
            appearance_features.append(f"Negated Change: {token.head.text} {token.text}")

    return appearance_features

def extract_knowledge(text):
    doc = nlp(text)
    knowledge_features = []
    
    # Define verbs related to knowledge
    knowledge_verbs = {"know", "understand", "believe", "think", "see", "hear", "feel", "notice", "say", "tell", "report"}
    
    for token in doc:
        # Check if token is a verb related to knowledge
        if token.pos_ == "VERB" and token.lemma_ in knowledge_verbs:
            knowledge_features.append(token.text)
        
        # Capture clauses starting with conjunctions related to knowledge
        if token.dep_ in ["mark"] and token.text.lower() in ["that"]:
            # Extract the clause following the conjunction
            clause = ' '.join([t.text for t in token.subtree])
            knowledge_features.append(f"Clause: {clause}")

        # Capture direct objects of knowledge verbs
        elif token.dep_ == "dobj" and token.head.pos_ == "VERB" and token.head.lemma_ in knowledge_verbs:
            knowledge_features.append(f"Object: {token.head.text} {token.text}")

    return knowledge_features

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
# print(mode_features)  # Output: ['must']

# # INTENTION
# text = "I want to go to the beach tomorrow."
# intention_features = extract_intention(text)
# print(intention_features)  # Output: ['want', 'Infinitive: go', 'Purpose: want']

# # RESULT
# text = "He had finished his work before the deadline."
# result_features = extract_result(text)
# print(result_features)  # Output: ['Perfect Tense: finished']

# # MANNER
# text = "He ran very quickly."
# manner_features = extract_manner(text)
# print(manner_features)  # Output: ['quickly', 'very quickly']

# # ASPECT
# text = "I have already finished my homework."
# aspect_features = extract_aspect(text)
# print(aspect_features)  # Output: ['have', 'already', 'finished']

# # STATUS
# text = "He did not go to the party."
# status_features = extract_status(text)
# print(status_features)  # Output: ['not', 'did not']

# # APPEARANCE
# text = "He wore a suit which made him look professional."
# appearance_features = extract_appearance(text)
# print(appearance_features)

# # KNOWLEDGE
# text = "She knows that the project is complete"
# knowledge_features = extract_knowledge(text)
# print(knowledge_features)