import spacy
import pandas as pd

nlp = spacy.load("en_core_web_md")

def extract_mode(text):
    doc = nlp(text)
    has_mode = False

    for token in doc:
        # Check for modal verbs or auxiliary verbs indicating mode
        if token.pos_ == "AUX" or (token.pos_ == "VERB" and token.lemma_ in ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]):
            has_mode = True
            break

        # Check for adverbs indicating possibility or necessity
        elif token.pos_ == "ADV" and token.lemma_ in ["necessarily", "probably", "possibly", "perhaps", "certainly"]:
            has_mode = True
            break

        # Check if the main verb (root) is at the beginning of the sentence, indicating modality
        elif token.pos_ == "VERB" and token.dep_ == "ROOT" and token.i == 0:
            has_mode = True
            break

    return 1 if has_mode else 0

def extract_intention(text):
    doc = nlp(text)
    has_intention = False

    for token in doc:
        # Check for verbs indicating intention
        if token.pos_ == "VERB" and token.lemma_ in ["want", "need", "desire", "intend", "wish", "plan", "aim", "decide", "hope"]:
            has_intention = True
            break

        # Check for "to" followed by a verb (infinitive form)
        elif token.text == "to" and token.head.pos_ == "VERB" and token.dep_ == "aux":
            has_intention = True
            break

        # Check for conjunctions indicating purpose with a verb
        elif token.text in ["so", "that", "order", "as"] and token.dep_ in ["mark", "advmod"]:
            if token.head.dep_ == "advcl" and token.head.head.pos_ == "VERB":
                has_intention = True
                break

        # Check for auxiliary verbs indicating future intention
        elif token.pos_ == "AUX" and token.lemma_ in ["will", "shall"]:
            has_intention = True
            break

    return 1 if has_intention else 0

def extract_result(text):
    doc = nlp(text)
    has_result = False

    for token in doc:   
        # Check for auxiliary verbs with past participle indicating result
        if token.lemma_ in ["have", "has", "had"] and token.dep_ == "aux" and token.head.tag_ == "VBN":
            has_result = True
            break

        # Check for verbs with adjectives or particles indicating result
        if token.pos_ == "VERB":
            for child in token.children:
                if child.pos_ in ["ADJ", "PART"]:
                    has_result = True
                    break
                
                elif child.pos_ == "PROPN" and (child.dep_ in ["attr", "dobj", "acomp", "oprd"] or child.head == token):
                    has_result = True
                    break

        # Check for adverbs indicating result
        if token.pos_ == "ADV" and token.lemma_ in ["so", "therefore", "thus", "hence"]:
            if token.dep_ == "cc" or token.head.dep_ == "conj":
                has_result = True
                break

        # Check for subordinating conjunctions indicating result
        elif token.pos_ == "SCONJ" and token.lemma_ in ["after", "because", "since", "as"]:
            has_result = True
            break
        
        # Check if the phrase "as a result" is in the text
        if "as a result" in text:
            has_result = True
            break

    return 1 if has_result else 0

def extract_manner(text):
    doc = nlp(text)
    has_manner = False

    for token in doc:
        # Check for adverbs (ADV) used as adverbial modifiers (advmod)
        if token.pos_ == "ADV" and token.dep_ == "advmod":
            has_manner = True
            break

    # If no manner found, check for prepositions (ADP) with objects
    if not has_manner:
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ in ["pobj", "obj"]:
                        has_manner = True
                        break
                if has_manner:
                    break

    # If still no manner found, check for adverbs (ADV) modifying other adverbs
    if not has_manner:
        for token in doc:
            if token.pos_ == "ADV" and token.dep_ == "advmod":
                if token.head.pos_ == "ADV":
                    has_manner = True
                    break

    return 1 if has_manner else 0

def extract_aspect(text):
    doc = nlp(text)
    has_aspect = False

    for token in doc:
        # Check for specific verbs indicating aspect
        if token.pos_ == "VERB" and token.lemma_ in ["start", "finish", "continue", "begin", "stop"]:
            has_aspect = True
            break

        # Check for auxiliary verbs with past participles indicating aspect
        if token.lemma_ in ["have", "has", "had"] and token.dep_ == "aux" and token.head.tag_ == "VBN":
            has_aspect = True
            break

        # Check for auxiliary verbs with gerunds indicating aspect
        if token.lemma_ in ["be"] and token.dep_ == "aux" and token.head.tag_.startswith("VBG"):
            has_aspect = True
            break

        # Check for adverbs indicating aspect
        if token.pos_ == "ADV" and token.lemma_ in ["already", "yet", "still"]:
            has_aspect = True
            break

        # Check for auxiliary verbs indicating future or continuous aspect
        if token.pos_ == "AUX" and token.lemma_ in ["will", "have", "be"]:
            has_aspect = True
            break

    return 1 if has_aspect else 0

def extract_status(text):
    doc = nlp(text)
    has_status = False
    
    # List of multi-word phrases indicating negation or status
    multi_word_negations = [
        "no longer", "not at all", "never again", "not really", "not yet", "not sure", "don't know"
    ]

    # Check if any multi-word negation phrases are present in the text
    for phrase in multi_word_negations:
        if phrase in text:
            has_status = True
            break

    # If no multi-word negations, check for single-word negations and dependency-based negations
    if not has_status:
        for token in doc:
            # Check for single-word negations
            if token.lemma_ in ["not", "never", "no"]:
                has_status = True
                break

            # Check for negation dependency with verbs or auxiliaries
            if token.dep_ == "neg" and token.head.pos_ in ["AUX", "VERB"]:
                has_status = True
                break

            # Check for general negation dependency
            if token.dep_ == "neg":
                has_status = True
                break

    return 1 if has_status else 0

def extract_appearance(text):
    doc = nlp(text)
    has_appearance = False

    for token in doc:
        # Check for coordinating conjunctions (CCONJ) or specific pronouns
        if token.pos_ == "CCONJ" or (token.pos_ == "PRON" and token.lemma_ in ["which", "that"]):
            has_appearance = True
            break

    # If no appearance found, check for verbs indicating change
    if not has_appearance:
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in ["become", "turn", "transform", "change"]:
                has_appearance = True
                break

    # If still no appearance found, check for negation with verbs indicating change
    if not has_appearance:
        for token in doc:
            if token.dep_ == "neg" and token.head.pos_ == "VERB" and token.head.lemma_ in ["become", "turn", "transform", "change"]:
                has_appearance = True
                break

    return 1 if has_appearance else 0

def extract_knowledge(text):
    doc = nlp(text)
    has_knowledge = False

    # Set of verbs associated with knowledge and perception
    knowledge_verbs = {"know", "realize", "remember", "learn", "recognize", "understand", "believe", "think", "see", "hear", "feel", "notice", "say", "tell", "inform", "report", "observe"}
    
    for token in doc:
        # Check for verbs associated with knowledge
        if token.pos_ == "VERB" and token.lemma_ in knowledge_verbs:
            has_knowledge = True
            break

        # Check for the conjunction "that" as a marker for knowledge
        if token.dep_ == "mark" and token.text.lower() in ["that"]:
            has_knowledge = True
            break

        # Check for direct objects (dobj) of verbs associated with knowledge
        if token.dep_ == "dobj" and token.head.pos_ == "VERB" and token.head.lemma_ in knowledge_verbs:
            has_knowledge = True
            break

    return 1 if has_knowledge else 0

def extract_description(text):
    doc = nlp(text)
    has_description = False

    for token in doc:
        # Check for verbs associated with description
        if token.pos_ == "VERB" and token.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True
            break

        # Check for complement clauses (ccomp) with description verbs
        if token.dep_ == "ccomp" and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True
            break

        # Check for direct speech with description verbs (ROOT position)
        if token.pos_ == "VERB" and token.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"] and token.dep_ == "ROOT":
            for child in token.children:
                if child.pos_ == "NOUN" and child.text.startswith('"'):
                    has_description = True
                    break

        # Check for complement clauses (ccomp) with specific verbs in ROOT position
        if token.pos_ == "VERB" and token.lemma_ in ["tell", "inform", "narrate"] and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "ccomp":
                    has_description = True
                    break

        # Check for description verbs with complement or open clausal complements
        if (token.dep_ == "ccomp" or token.dep_ == "xcomp") and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True
            break

        # Check for adjectives or adverbs modifying description verbs
        if (token.dep_ == "amod" or token.dep_ == "advmod") and token.head.pos_ == "VERB" and token.head.lemma_ in ["say", "tell", "explain", "describe", "report", "narrate", "inform"]:
            has_description = True
            break

    return 1 if has_description else 0

def extract_supposition(text):
    doc = nlp(text)
    has_supposition = False

    for token in doc:
        # Check for modal verbs indicating supposition
        if token.lemma_ in ["will", "would", "might", "may", "could", "should"]:
            has_supposition = True
            break

        # Check for subordinating conjunction "if" indicating conditionality
        if token.pos_ == "SCONJ" and token.lemma_ == "if":
            has_supposition = True
            break

        # Check for verbs related to expectation or prediction
        if token.pos_ == "VERB" and token.lemma_ in ["expect", "predict", "assume", "suppose", "anticipate"]:
            has_supposition = True
            break

        # Check for adverbs indicating probability or possibility
        if token.pos_ == "ADV" and token.lemma_ in ["probably", "possibly", "maybe", "likely"]:
            has_supposition = True
            break

         # Check for dependency labels that may indicate supposition
        if token.dep_ in ["aux", "advmod", "ccomp"]:
            has_supposition = True
            break

    return 1 if has_supposition else 0

def extract_subjectivation(text):
    doc = nlp(text)
    has_subjectivation = False

    for token in doc:
        # Check for pronouns indicating subjectivity
        if token.pos_ == "PRON" and token.lemma_.lower() in ["i", "you", "he", "she", "it", "we", "they"]:
            has_subjectivation = True
            break

        # Check for verbs related to thinking or perceiving with ROOT dependency
        if token.pos_ == "VERB" and token.lemma_ in ["think", "believe", "feel", "perceive", "consider"]:
            has_subjectivation = True
            if token.dep_ == "ROOT":
                has_subjectivation = True
                break

        # Check for ROOT verbs with "VBZ" tag and subject pronouns
        if token.pos_ == "VERB" and token.dep_ == "ROOT" and token.tag_ == "VBZ":
            for child in token.children:
                if child.dep_ == "nsubj" and child.pos_ == "PRON":
                    has_subjectivation = True
                    break

        # Check for adjectives in complement or modifier positions related to subjectivity
        if token.pos_ == "ADJ" and token.dep_ == "ccomp":
            has_subjectivation = True
            break

        # Check if an adjective modifying a pronoun indicates subjectivation
        if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ == "PRON":
            has_subjectivation = True
            break

        # Check for subject and clausal subject dependencies
        if token.dep_ in ["nsubj", "csubj"]:
            has_subjectivation = True
            break

    return 1 if has_subjectivation else 0

def extract_attitude(text):
    doc = nlp(text)
    has_attitude = False

    # Define sets of emotion-related verbs and adjectives
    emotion_verbs = {
        "feel", "love", "hate", "enjoy", "fear", "worry", 
        "regret", "like", "dislike", "admire", "appreciate", 
        "resent", "cherish", "despise", "adore", "savor", 
        "lament", "yearn", "long", "speak", "disappoint"}

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
        # Check if the verb indicates an emotional state
        if token.pos_ == "VERB" and token.lemma_ in emotion_verbs:
            has_attitude = True
            break
        
        # Check if the adjective indicates an emotional state
        if token.pos_ == "ADJ" and token.lemma_ in emotion_adjectives:
            has_attitude = True
            break
        
        # Check if the adverb modifies a verb indicating emotion
        if token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "VERB" and token.head.lemma_ in emotion_verbs:
            has_attitude = True
            break
        
        # Check if a verb related to perception is modifying an adjective
        if token.pos_ == "VERB" and token.lemma_ in ["see", "hear", "feel"] and token.head.pos_ == "ADJ":
            has_attitude = True
            break
        
        # Check for interjections indicating emotional reactions
        if token.pos_ == "INTJ":
            has_attitude = True
            break
        
        # Check if an emotional verb has a subject
        if token.pos_ == "VERB" and token.lemma_ in emotion_verbs:
            for child in token.children:
                if child.dep_ == "nsubj":
                    has_attitude = True
                    break

        # Check if an adjective modifying a subject indicates attitude
        if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.dep_ == "nsubj":
            has_attitude = True
            break

        # Check for various dependency labels indicating emotional content
        if token.dep_ in ["nsubj", "amod", "advmod"]:
            has_attitude = True
            break

    return 1 if has_attitude else 0

def extract_comparative(text):
    doc = nlp(text)
    has_comparative = False

    # Define comparative and superlative phrases and words
    comparative_phrases = [
        "than", "compared to", "in comparison with", "versus", "in relation to",
        "as opposed to", "more than", "less than", "greater than", "smaller than",
        "better than", "worse than", "superior to", "inferior to", "like", "unlike",
        "rather than", "instead of"
    ]
    
    comparative_words = {"more", "less", "better", "worse"}
    superlative_words = {"most", "least", "best", "worst"}

    for token in doc:
        # Check if the token is an adjective or adverb with comparative suffix
        if (token.pos_ == "ADJ" or token.pos_ == "ADV") and token.lemma_.endswith("er"):
            has_comparative = True
            break

        # Check if the token is an adjective or adverb with superlative suffix
        if (token.pos_ == "ADJ" or token.pos_ == "ADV") and token.lemma_.endswith("est"):
            has_comparative = True
            break

        # Check if the token is a known superlative word
        if token.text.lower() in superlative_words:
            has_comparative = True
            break

        # Check if the token is a known comparative word
        if token.text.lower() in comparative_words:
            has_comparative = True
            break

        # Check if the token is part of a comparative phrase
        if token.text.lower() in comparative_phrases:
            has_comparative = True
            break

        # Check if an adjective or adverb modifying another adjective or adverb indicates comparison
        if token.dep_ in ["amod", "advmod"]:
            if token.head.pos_ in ["ADJ", "ADV"]:
                # Check if the head token is a comparative or superlative word
                if token.head.lemma_.endswith("er") or token.head.text.lower() in comparative_words:
                    has_comparative = True
                    break
                elif token.head.lemma_.endswith("est") or token.head.text.lower() in superlative_words:
                    has_comparative = True
                    break

    return 1 if has_comparative else 0

def extract_quantifier(text):
    doc = nlp(text)
    has_quantifier = False

    # Define expressions and phrases that indicate quantity or proportion
    degree_expressions = ["a lot of", "a little", "enough", "plenty of"]
    proportional_phrases = ["half", "most", "majority of", "part of", "fraction of"]

    for token in doc:
        # Check if the token is a determiner (DET) or adjective (ADJ) that denotes quantity
        if (token.pos_ == "DET" or token.pos_ == "ADJ") and token.lemma_ in ["all", "some", "many", "few", "several", "much", "little", "none"]:
            has_quantifier = True
            break

        # Check if the token is a numeral (NUM), which indicates a quantity
        if token.pos_ == "NUM":
            has_quantifier = True
            break

        # Check if the token's subtree forms a known degree expression
        if token.text in ["a", "lot", "little", "plenty", "majority"]:
            span = " ".join([w.text for w in token.subtree])
            if span in degree_expressions:
                has_quantifier = True
                break

        # Check if the token is part of a proportional phrase
        if token.text in proportional_phrases:
            has_quantifier = True
            break

        # Check if the token is part of a quantity modifier (nummod) or determiner (det)
        if token.dep_ in ["nummod", "det"]:
            has_quantifier = True
            break

        # Check if the token is an adverb indicating an approximate quantity
        if token.pos_ == "ADV" and token.lemma_ in ["almost", "nearly", "approximately", "about"]:
            has_quantifier = True
            break

    return 1 if has_quantifier else 0

def extract_qualification(text):
    doc = nlp(text)
    has_qualification = False

    for token in doc:
        # Check if the token is an adjective (ADJ) that modifies a noun (amod)
        if token.pos_ == "ADJ" and token.dep_ == "amod":
            has_qualification = True
            break

        # Check if the token is an adverb (ADV) modifying an adjective (advmod)
        if token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "ADJ":
            has_qualification = True
            break

        # Check if the token is an adjective (ADJ) modifying a noun (amod)
        if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ == "NOUN":
            has_qualification = True
            break

        # Check if the token is an adjective (ADJ) with a tag indicating past participle or gerund
        if token.pos_ == "ADJ" and token.tag_ in {"VBN", "VBG", "VBP"}:
            has_qualification = True
            break

        # Check if the token is an adjective or adverb in a modifying relation (amod, advmod)
        if token.dep_ in ["amod", "advmod"]:
            has_qualification = True
            break

        # Check if the token is a relative clause modifier (relcl)
        if token.dep_ == "relcl":
            has_qualification = True
            break

    return 1 if has_qualification else 0

def extract_explanation(text):
    doc = nlp(text)
    has_explanation = False

    explanatory_conjunctions = ["because", "since", "therefore", "so"]
    explicative_phrases = ["in other words", "namely"]

    for token in doc:
        # Check if the token is part of a relative clause (acl) or a relative clause (relcl)
        if token.dep_ in ["acl", "relcl"]:
            span = list(token.subtree)
            has_explanation = True
            break

        # Check if the token is a punctuation mark that could indicate a parenthetical explanation
        if token.dep_ == "punct" and token.text in ["(", ")"]:
            parenthetical_span = list(token.subtree)
            if len(parenthetical_span) > 1:
                has_explanation = True
                break

        # Check if the token is a subordinating conjunction (SCONJ) used for explanations
        if token.pos_ == "SCONJ" and token.lemma_ in explanatory_conjunctions:
            has_explanation = True
            break

        # Check if the token is part of an appositive phrase (appos)
        if token.dep_ == "appos":
            span = list(token.subtree)
            has_explanation = True
            break

        # Check if the token is part of an explicative phrase
        if token.text.lower() in explicative_phrases:
            has_explanation = True
            break

    return 1 if has_explanation else 0

def create_feature_vector(text):
    mode_feature = extract_mode(text)
    intention_feature = extract_intention(text)
    result_feature = extract_result(text)
    manner_feature = extract_manner(text)
    aspect_feature = extract_aspect(text)
    status_feature = extract_status(text)
    appearance_feature = extract_appearance(text)
    knowledge_feature = extract_knowledge(text)
    description_feature = extract_description(text)
    supposition_feature = extract_supposition(text)
    subjectivation_feature = extract_subjectivation(text)
    attitude_feature = extract_attitude(text)
    comparative_feature = extract_comparative(text)
    quantifier_feature = extract_quantifier(text)
    qualification_feature = extract_qualification(text)
    explanation_feature = extract_explanation(text)
    return [
        mode_feature, intention_feature, result_feature, manner_feature,
        aspect_feature, status_feature, appearance_feature, knowledge_feature,
        description_feature, supposition_feature, subjectivation_feature, attitude_feature,
        comparative_feature, quantifier_feature, qualification_feature, explanation_feature
        ]

def extract_eng_features_from_dataframe(df):
    df['features'] = df['sentence'].apply(create_feature_vector)

    features_df = pd.DataFrame(df['features'].tolist(), columns=[
        'mode', 'intention', 'result', 'manner',
        'aspect', 'status', 'appearance', 'knowledge',
        'description', 'supposition', 'subjectivation', 'attitude',
        'comparative', 'quantifier', 'qualification', 'explanation'
    ])

    return features_df