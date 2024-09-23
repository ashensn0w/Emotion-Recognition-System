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

# # Resultative constructions <- to fix
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

def extract_description(text):
    doc = nlp(text)
    has_description = False  # Initialize flag for description features

    for token in doc:
        # Reporting verbs
        if token.pos_ == "VERB" and token.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True

        # Speech or thought clauses
        if token.dep_ == "ccomp" and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True

        # Quoted speech or dialogue
        if token.pos_ == "VERB" and token.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"] and token.dep_ == "ROOT":
            for child in token.children:
                if child.pos_ == "NOUN" and child.text.startswith('"'):
                    has_description = True

        # Indirect speech
        if token.pos_ == "VERB" and token.lemma_ in ["tell", "inform", "narrate"] and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "ccomp":
                    has_description = True

        # Dependency relations
        if (token.dep_ == "ccomp" or token.dep_ == "xcomp") and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True

        # Modifier relations
        if (token.dep_ == "amod" or token.dep_ == "advmod") and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True

    return 1 if has_description else 0

def extract_supposition(text):
    doc = nlp(text)
    has_supposition = False  # Initialize flag for supposition features

    for token in doc:
        # Modal verbs indicating uncertainty or future possibility
        if token.lemma_ in ["will", "would", "might", "may", "could", "should"]:
            has_supposition = True

        # Conditional sentences (e.g., "if" as subordinating conjunction)
        if token.pos_ == "SCONJ" and token.lemma_ == "if":
            has_supposition = True

        # Verbs of prediction or expectation
        if token.pos_ == "VERB" and token.lemma_ in ["expect", "predict", "assume", "suppose", "anticipate"]:
            has_supposition = True

        # Epistemic adverbs and phrases (indicating likelihood or uncertainty)
        if token.pos_ == "ADV" and token.lemma_ in ["probably", "possibly", "maybe", "likely"]:
            has_supposition = True

        # Dependency relations (Auxiliary verbs or adverbs related to modality)
        if token.dep_ in ["aux", "advmod", "ccomp"]:
            has_supposition = True

    return 1 if has_supposition else 0

# # Subjective adjectives (adjectives modifying pronouns) <- to fix
def extract_subjectivation(text):
    doc = nlp(text)
    has_subjectivation = False

    for token in doc:
        # Personal pronouns
        if token.pos_ == "PRON" and token.lemma_.lower() in ["i", "you", "he", "she", "it", "we", "they"]:
            has_subjectivation = True

        # Cognitive verbs related to perception
        if token.pos_ == "VERB" and token.lemma_ in ["think", "believe", "feel", "perceive", "consider"]:
            has_subjectivation = True
            # Sentences expressing judgments or opinions (root verbs)
            if token.dep_ == "ROOT":
                has_subjectivation = True

        # Subject-verb agreement (more general than just checking "He")
        if token.pos_ == "VERB" and token.dep_ == "ROOT" and token.tag_ == "VBZ":
            for child in token.children:
                if child.dep_ == "nsubj" and child.pos_ == "PRON":
                    has_subjectivation = True

        # Subjective adjectives (adjectives modifying pronouns)
        if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ == "PRON":
            has_subjectivation = True

        # Dependency relations involving subject
        if token.dep_ in ["nsubj", "csubj"]:
            has_subjectivation = True

    return 1 if has_subjectivation else 0

# # Adverbial modifiers of emotional verbs AND # Dependency relations <- to fix
def extract_attitude(text):
    doc = nlp(text)
    has_attitude = False

# Define verbs related to emotions or attitudes
    emotion_verbs = {
        "feel", "love", "hate", "enjoy", "fear", "worry", 
        "regret", "like", "dislike", "admire", "appreciate", 
        "resent", "cherish", "despise", "adore", "savor", 
        "lament", "yearn", "long"}

    # Define adjectives indicating emotions or attitudes
    emotion_adjectives = {
        "happy", "sad", "angry", "excited", "anxious", 
        "disappointed", "elated", "frustrated", "content", 
        "nervous", "guilty", "hopeful", "relieved", 
        "pleased", "joyful", "upset", "bored", 
        "embarrassed", "pessimistic", "optimistic", 
        "euphoric", "distraught", "jubilant", 
        "melancholic", "overjoyed"
    }
    
    for token in doc:
        # Emotion or psychological verbs
        if token.pos_ == "VERB" and token.lemma_ in emotion_verbs:
            has_attitude = True
        
        # Adjectives indicating emotions or attitudes
        if token.pos_ == "ADJ" and token.lemma_ in emotion_adjectives:
            has_attitude = True
        
        # Adverbial modifiers of emotional verbs
        if token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "VERB" and token.head.lemma_ in emotion_verbs:
            has_attitude = True
        
        # Perception or sensory verbs with emotion (e.g., feel + adjective)
        if token.pos_ == "VERB" and token.lemma_ in ["see", "hear", "feel"] and token.head.pos_ == "ADJ":
            has_attitude = True
        
        # Exclamations or interjections
        if token.pos_ == "INTJ":
            has_attitude = True
        
        # Dependency relations
        if token.dep_ in ["nsubj", "amod", "advmod"]:
            has_attitude = True

    return 1 if has_attitude else 0

def extract_comparative(text):
    doc = nlp(text)
    has_comparative = False

    # Define a more comprehensive list of comparative and superlative phrases
    comparative_phrases = [
        "than", "compared to", "in comparison with", "versus", "in relation to",
        "as opposed to", "more than", "less than", "greater than", "smaller than",
        "better than", "worse than", "superior to", "inferior to", "like", "unlike",
        "rather than", "instead of"
    ]
    
    # Define comparative and superlative words
    comparative_words = {"more", "less", "better", "worse"}
    superlative_words = {"most", "least", "best", "worst"}

    for token in doc:
        # Comparative adjectives and adverbs
        if (token.pos_ == "ADJ" or token.pos_ == "ADV") and token.lemma_.endswith("er"):
            has_comparative = True

        # Superlative adjectives and adverbs
        if (token.pos_ == "ADJ" or token.pos_ == "ADV") and token.lemma_.endswith("est"):
            has_comparative = True

        # Superlative words
        if token.text.lower() in superlative_words:
            has_comparative = True

        # Comparative words
        if token.text.lower() in comparative_words:
            has_comparative = True

        # Comparative constructions
        if token.text.lower() in comparative_phrases:
            has_comparative = True

        # Dependency relations related to comparatives
        if token.dep_ in ["amod", "advmod"]:
            # Check if the head token is comparative or superlative
            if token.head.pos_ in ["ADJ", "ADV"]:
                if token.head.lemma_.endswith("er") or token.head.text.lower() in comparative_words:
                    has_comparative = True
                elif token.head.lemma_.endswith("est") or token.head.text.lower() in superlative_words:
                    has_comparative = True

    return 1 if has_comparative else 0

def extract_quantifier(text):
    doc = nlp(text)
    has_quantifier = False

    # Define lists for degree expressions and proportional phrases
    degree_expressions = ["a lot of", "a little", "enough", "plenty of"]
    proportional_phrases = ["half", "most", "majority of", "part of", "fraction of"]

    for token in doc:
        # Quantifiers (determiners)
        if (token.pos_ == "DET" or token.pos_ == "ADJ") and token.lemma_ in ["all", "some", "many", "few", "several", "much", "little", "none"]:
            has_quantifier = True

        # Numerical expressions
        if token.pos_ == "NUM":
            has_quantifier = True

        # Expressions of degree (multi-word expressions)
        if token.text in ["a", "lot", "little", "plenty", "majority"]:
            span = " ".join([w.text for w in token.subtree])
            if span in degree_expressions:
                has_quantifier = True

        # Proportional phrases (multi-word expressions)
        if token.text in proportional_phrases:
            has_quantifier = True

        # Dependency relations related to quantifiers
        if token.dep_ in ["nummod", "det"]:
            has_quantifier = True

        # Adverbs indicating quantification
        if token.pos_ == "ADV" and token.lemma_ in ["almost", "nearly", "approximately", "about"]:
            has_quantifier = True

    return 1 if has_quantifier else 0

def extract_qualification(text):
    doc = nlp(text)
    has_qualification = False

    for token in doc:
        # Qualifying adjectives
        if token.pos_ == "ADJ" and token.dep_ == "amod":
            has_qualification = True

        # Intensifying adverbs
        if token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "ADJ":
            has_qualification = True

        # Adjectival phrases
        if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ == "NOUN":
            has_qualification = True

        # Participial adjectives
        if token.pos_ == "ADJ" and token.tag_ in {"VBN", "VBG", "VBP"}:
            has_qualification = True

        # Dependency relations
        if token.dep_ in ["amod", "advmod"]:
            has_qualification = True

        # Qualifying clauses (relative clauses)
        if token.dep_ == "relcl":
            has_qualification = True

    return 1 if has_qualification else 0

def extract_explanation(text):
    doc = nlp(text)
    has_explanation = False

    # Define phrases and conjunctions related to explanations
    explanatory_conjunctions = ["because", "since", "therefore", "so"]
    explicative_phrases = ["in other words", "namely"]

    # Check for explanatory clauses
    for token in doc:
        if token.dep_ in ["acl", "relcl"]:
            span = list(token.subtree)
            has_explanation = True

        # Check for parenthetical phrases
        if token.dep_ == "punct" and token.text in ["(", ")"]:
            parenthetical_span = list(token.subtree)
            if len(parenthetical_span) > 1:  # Ensure there's content within parentheses
                has_explanation = True

        # Check for explanatory conjunctions
        if token.pos_ == "SCONJ" and token.lemma_ in explanatory_conjunctions:
            has_explanation = True

        # Check for appositive phrases
        if token.dep_ == "appos":
            span = list(token.subtree)
            has_explanation = True

        # Check for explicative phrases
        if token.text.lower() in explicative_phrases:
            has_explanation = True

    return 1 if has_explanation else 0

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

# # DESCRIPTION
# text = "John told Sarah about the new plan."
# description_features = extract_description(text)
# print(description_features)

# # SUPPOSITION
# text = "If I say I'm sorry, would you forgive me?"
# supposition_features = extract_supposition(text)
# print(supposition_features)

# # SUBJECTIVATION
# text = "He feels the situation is under control."
# subjectivation_features = extract_subjectivation(text)
# print(subjectivation_features)

# # ATTITUDE
# text = "She was disappointed by the news."
# attitude_features = extract_attitude(text)
# print(attitude_features)

# # COMPARATIVE
# text = "The new restaurant is much better than the old one."
# comparative_features = extract_comparative(text)
# print(comparative_features)

# # QUANTIFIER
# text = "She almost finished the task."
# quantifier_features = extract_quantifier(text)
# print(quantifier_features)

# # QUALIFICATION
# text = "The book, which was published last year, is a bestseller."
# qualification_features = extract_qualification(text)
# print(qualification_features)

# # EXPLANATION
# text = "He is an excellent chef, namely, a master of French cuisine."
# explanation_features = extract_explanation(text)
# print(explanation_features)