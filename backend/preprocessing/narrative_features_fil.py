import spacy

# Load the language model
nlp = spacy.load("xx_ent_wiki_sm")

def extract_mode(text):
    doc = nlp(text)
    
    # Define keywords for each mode
    mode_keywords = {
        'Possibility': ['maaari', 'pwedeng'],
        'Impossibility': ['hindi', 'walang'],
        'Necessity': ['dapat', 'kailangan'],
        'Prohibition': ['bawal']
    }
    
    detected_modes = {
        'Possibility': False,
        'Impossibility': False,
        'Necessity': False,
        'Prohibition': False
    }
    
    # Check for mode keywords in the text
    for token in doc:
        if token.text.lower() in mode_keywords['Possibility']:
            detected_modes['Possibility'] = True
        elif token.text.lower() in mode_keywords['Impossibility']:
            detected_modes['Impossibility'] = True
        elif token.text.lower() in mode_keywords['Necessity']:
            detected_modes['Necessity'] = True
        elif token.text.lower() in mode_keywords['Prohibition']:
            detected_modes['Prohibition'] = True
        
        # Additional dependency parsing checks for modal verbs
        if token.dep_ in ['acomp', 'xcomp', 'ccomp'] and token.head.text.lower() in mode_keywords['Possibility']:
            detected_modes['Possibility'] = True
        elif token.dep_ in ['acomp', 'xcomp', 'ccomp'] and token.head.text.lower() in mode_keywords['Impossibility']:
            detected_modes['Impossibility'] = True

    return 1 if any(detected_modes.values()) else 0


def extract_intention(text):
    doc = nlp(text)
    
    # Define keywords for each intention feature
    intention_keywords = {
        'Infinitive Verbs': ['mag-aaral', 'pumunta'],
        'Modal Verbs': ['nais', 'gusto', 'hangad'],
        'Purpose Clauses': ['upang', 'para sa'],
        'Auxiliary Verbs': ['babalik', 'magiging']
    }
    
    detected_intentions = {
        'Infinitive Verbs': False,
        'Modal Verbs': False,
        'Purpose Clauses': False,
        'Auxiliary Verbs': False
    }
    
    # Check for intention keywords in the text
    for token in doc:
        if token.text.lower() in intention_keywords['Infinitive Verbs']:
            detected_intentions['Infinitive Verbs'] = True
        elif token.text.lower() in intention_keywords['Modal Verbs']:
            detected_intentions['Modal Verbs'] = True
        elif token.text.lower() in intention_keywords['Purpose Clauses']:
            detected_intentions['Purpose Clauses'] = True
        elif token.text.lower() in intention_keywords['Auxiliary Verbs']:
            detected_intentions['Auxiliary Verbs'] = True
        
        # Additional dependency parsing checks for intention-related verbs
        if token.dep_ in ['acl', 'amod', 'xcomp'] and token.head.text.lower() in intention_keywords['Modal Verbs']:
            detected_intentions['Modal Verbs'] = True
        elif token.dep_ in ['acl', 'xcomp'] and token.text.lower() in intention_keywords['Infinitive Verbs']:
            detected_intentions['Infinitive Verbs'] = True

    return 1 if any(detected_intentions.values()) else 0


def extract_result(text):
    doc = nlp(text)
    
    # Define keywords for result feature
    result_keywords = {
        'Completed Actions': ['natapos', 'nagawa', 'nakuha'],
        'Perfect Aspect Verbs': ['nagkaroon', 'nagawa']
    }
    
    detected_results = {
        'Completed Actions': False,
        'Perfect Aspect Verbs': False
    }
    
    # Check for result keywords in the text
    for token in doc:
        if token.text.lower() in result_keywords['Completed Actions']:
            detected_results['Completed Actions'] = True
        elif token.text.lower() in result_keywords['Perfect Aspect Verbs']:
            detected_results['Perfect Aspect Verbs'] = True
        
        # Additional dependency parsing checks for result-related verbs
        if token.dep_ in ['attr', 'ccomp', 'acomp'] and token.head.text.lower() in result_keywords['Completed Actions']:
            detected_results['Completed Actions'] = True
        elif token.dep_ in ['attr', 'xcomp'] and token.text.lower() in result_keywords['Perfect Aspect Verbs']:
            detected_results['Perfect Aspect Verbs'] = True

    return 1 if any(detected_results.values()) else 0


def extract_manner(text):
    doc = nlp(text)
    
    # Define keywords for manner feature
    manner_keywords = {
        'Adverbs': ['maayos', 'mabilis', 'mahigpit'],  # Add more as needed
        'Adjectives as Adverbs': ['maganda']  # Add more as needed
    }
    
    detected_manners = {
        'Adverbs': False,
        'Adjectives as Adverbs': False
    }
    
    # Check for manner keywords in the text
    for token in doc:
        if token.text.lower() in manner_keywords['Adverbs']:
            detected_manners['Adverbs'] = True
        elif token.text.lower() in manner_keywords['Adjectives as Adverbs']:
            detected_manners['Adjectives as Adverbs'] = True
        
        # Additional dependency parsing checks for manner
        if token.dep_ in ['advmod'] and token.text.lower() in manner_keywords['Adverbs']:
            detected_manners['Adverbs'] = True
        elif token.dep_ in ['amod'] and token.text.lower() in manner_keywords['Adjectives as Adverbs']:
            detected_manners['Adjectives as Adverbs'] = True

    return 1 if any(detected_manners.values()) else 0


def extract_aspect(text):
    doc = nlp(text)
    
    # Define keywords for aspect feature
    aspect_keywords = {
        'Aspectual Markers': ['nag-', 'naka-', 'nagsa-'],  # Add more as needed
        'Verbal Affixes': ['nag-aaral', 'natapos']  # Add more as needed
    }
    
    detected_aspects = {
        'Aspectual Markers': False,
        'Verbal Affixes': False
    }
    
    # Check for aspect keywords in the text
    for token in doc:
        if token.text.lower() in aspect_keywords['Aspectual Markers']:
            detected_aspects['Aspectual Markers'] = True
        elif token.text.lower() in aspect_keywords['Verbal Affixes']:
            detected_aspects['Verbal Affixes'] = True
        
        # Additional dependency parsing checks for aspect
        if token.dep_ in ['acl', 'amod'] and any(affix in token.text.lower() for affix in aspect_keywords['Aspectual Markers']):
            detected_aspects['Aspectual Markers'] = True
        elif token.dep_ in ['acl', 'xcomp'] and any(affix in token.text.lower() for affix in aspect_keywords['Verbal Affixes']):
            detected_aspects['Verbal Affixes'] = True

    return 1 if any(detected_aspects.values()) else 0


def extract_status(text):
    doc = nlp(text)
    
    # Define keywords for status feature
    status_keywords = {
        'Negation Words': ['hindi', 'wala', 'huwag']
    }
    
    detected_status = {
        'Negation Words': False
    }
    
    # Check for status keywords in the text
    for token in doc:
        if token.text.lower() in status_keywords['Negation Words']:
            detected_status['Negation Words'] = True
        
        # Additional dependency parsing checks for negation
        if token.dep_ in ['neg'] and token.text.lower() in status_keywords['Negation Words']:
            detected_status['Negation Words'] = True

    return 1 if any(detected_status.values()) else 0


def extract_appearance(text):
    doc = nlp(text)
    
    # Define keywords for appearance feature
    appearance_keywords = {
        'Transition Words': ['naging', 'pinalitan', 'nagbago']
    }
    
    detected_appearance = {
        'Transition Words': False
    }
    
    # Check for appearance keywords in the text
    for token in doc:
        if token.text.lower() in appearance_keywords['Transition Words']:
            detected_appearance['Transition Words'] = True
        
        # Additional dependency parsing checks for appearance
        if token.dep_ in ['ccomp', 'acl'] and token.text.lower() in appearance_keywords['Transition Words']:
            detected_appearance['Transition Words'] = True

    return 1 if any(detected_appearance.values()) else 0


def extract_knowledge(text):
    doc = nlp(text)
    
    # Define keywords for knowledge feature
    knowledge_keywords = {
        'Knowledge Verbs': ['alam', 'nagpapakita', 'nalaman']
    }
    
    detected_knowledge = {
        'Knowledge Verbs': False
    }
    
    # Check for knowledge keywords in the text
    for token in doc:
        if token.text.lower() in knowledge_keywords['Knowledge Verbs']:
            detected_knowledge['Knowledge Verbs'] = True
        
        # Additional dependency parsing checks for knowledge
        if token.dep_ in ['ccomp', 'acl'] and token.text.lower() in knowledge_keywords['Knowledge Verbs']:
            detected_knowledge['Knowledge Verbs'] = True

    return 1 if any(detected_knowledge.values()) else 0


def extract_description(text):
    doc = nlp(text)
    
    # Define keywords for description feature
    description_keywords = {
        'Descriptive Phrases': ['sinabi', 'nasabi', 'sinasabi']
    }
    
    detected_descriptions = {
        'Descriptive Phrases': False
    }
    
    # Check for description keywords in the text
    for token in doc:
        if token.text.lower() in description_keywords['Descriptive Phrases']:
            detected_descriptions['Descriptive Phrases'] = True
        
        # Additional dependency parsing checks for description
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in description_keywords['Descriptive Phrases']:
            detected_descriptions['Descriptive Phrases'] = True

    return 1 if any(detected_descriptions.values()) else 0

def extract_supposition(text):
    doc = nlp(text)
    
    # Define keywords for supposition feature
    supposition_keywords = {
        'Supposition Modal Verbs': ['maaaring', 'baka', 'sana']
    }
    
    detected_suppositions = {
        'Supposition Modal Verbs': False
    }
    
    # Check for supposition keywords in the text
    for token in doc:
        if token.text.lower() in supposition_keywords['Supposition Modal Verbs']:
            detected_suppositions['Supposition Modal Verbs'] = True
        
        # Additional dependency parsing checks for supposition
        if token.dep_ in ['xcomp', 'acl'] and token.text.lower() in supposition_keywords['Supposition Modal Verbs']:
            detected_suppositions['Supposition Modal Verbs'] = True

    return 1 if any(detected_suppositions.values()) else 0


def extract_subjectivation(text):
    doc = nlp(text)
    
    # Define keywords for subjectivation feature
    subjectivation_keywords = {
        'Perception Verbs': ['nakikita', 'nararamdaman', 'iniisip']
    }
    
    detected_subjectivation = {
        'Perception Verbs': False
    }
    
    # Check for subjectivation keywords in the text
    for token in doc:
        if token.text.lower() in subjectivation_keywords['Perception Verbs']:
            detected_subjectivation['Perception Verbs'] = True
        
        # Additional dependency parsing checks for subjectivation
        if token.dep_ in ['ccomp', 'xcomp'] and token.text.lower() in subjectivation_keywords['Perception Verbs']:
            detected_subjectivation['Perception Verbs'] = True

    return 1 if any(detected_subjectivation.values()) else 0


def extract_attitude(text):
    doc = nlp(text)
    
    # Define keywords for attitude feature
    attitude_keywords = {
        'Emotion-related Adjectives': ['masaya', 'nalungkot', 'nagulat']
    }
    
    detected_attitudes = {
        'Emotion-related Adjectives': False
    }
    
    # Check for attitude keywords in the text
    for token in doc:
        if token.text.lower() in attitude_keywords['Emotion-related Adjectives']:
            detected_attitudes['Emotion-related Adjectives'] = True
        
        # Additional dependency parsing checks for attitude
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in attitude_keywords['Emotion-related Adjectives']:
            detected_attitudes['Emotion-related Adjectives'] = True

    return 1 if any(detected_attitudes.values()) else 0


def extract_comparative(text):
    doc = nlp(text)
    
    # Define keywords for comparative feature
    comparative_keywords = {
        'Comparative Adjectives': ['mas', 'higit', 'kaysa']
    }
    
    detected_comparative = {
        'Comparative Adjectives': False
    }
    
    # Check for comparative keywords in the text
    for token in doc:
        if token.text.lower() in comparative_keywords['Comparative Adjectives']:
            detected_comparative['Comparative Adjectives'] = True
        
        # Additional dependency parsing checks for comparative
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in comparative_keywords['Comparative Adjectives']:
            detected_comparative['Comparative Adjectives'] = True

    return 1 if any(detected_comparative.values()) else 0


def extract_quantifier(text):
    doc = nlp(text)
    
    # Define keywords for quantifier feature
    quantifier_keywords = {
        'Quantifiers': ['marami', 'mas', 'ilan' , 'lahat']
    }
    
    detected_quantifiers = {
        'Quantifiers': False
    }
    
    # Check for quantifier keywords in the text
    for token in doc:
        if token.text.lower() in quantifier_keywords['Quantifiers']:
            detected_quantifiers['Quantifiers'] = True
        
        # Additional dependency parsing checks for quantifiers
        if token.dep_ in ['amod', 'nummod'] and token.text.lower() in quantifier_keywords['Quantifiers']:
            detected_quantifiers['Quantifiers'] = True

    return 1 if any(detected_quantifiers.values()) else 0


def extract_qualification(text):
    doc = nlp(text)
    
    # Define keywords for qualification feature
    qualification_keywords = {
        'Qualifying Adjectives/Adverbs': ['napaka', 'sobra', 'talaga']
    }
    
    detected_qualifications = {
        'Qualifying Adjectives/Adverbs': False
    }
    
    # Check for qualification keywords in the text
    for token in doc:
        if token.text.lower() in qualification_keywords['Qualifying Adjectives/Adverbs']:
            detected_qualifications['Qualifying Adjectives/Adverbs'] = True
        
        # Additional dependency parsing checks for qualification
        if token.dep_ in ['amod', 'advmod'] and token.text.lower() in qualification_keywords['Qualifying Adjectives/Adverbs']:
            detected_qualifications['Qualifying Adjectives/Adverbs'] = True

    return 1 if any(detected_qualifications.values()) else 0


def extract_explanation(text):
    doc = nlp(text)
    
    # Define keywords for explanation feature
    explanation_keywords = {
        'Explanation Phrases': ['dahil sa', 'upang', 'sapagkat']
    }
    
    detected_explanations = {
        'Explanation Phrases': False
    }
    
    # Check for explanation keywords in the text
    for token in doc:
        if token.text.lower() in explanation_keywords['Explanation Phrases']:
            detected_explanations['Explanation Phrases'] = True
        
        # Additional dependency parsing checks for explanation
        if token.dep_ in ['mark', 'ccomp'] and token.text.lower() in explanation_keywords['Explanation Phrases']:
            detected_explanations['Explanation Phrases'] = True

    return 1 if any(detected_explanations.values()) else 0



# -----------------------------------------------------------------------------------------------------
# MODE
text = "Maaari kong gawin ito"
mode_features = extract_mode(text)
print(mode_features)

# INTENTION
text = "Nais kong maging ako"
intention_features = extract_intention(text)
print(intention_features)

# RESULT
text = "Ang proyekto ay natapos na"
result_features = extract_result(text)
print(result_features)  

# MANNER
text = "Ang pagsayaw ng maayos ay maganda"
result_features = extract_manner(text)
print(result_features) 

# ASPECT
text = "Natapos ko na ang proyekto"
result_features = extract_aspect(text)
print(result_features) 

# STATUS
text = "Hindi ko magagawa ito."
result_features = extract_status(text)
print(result_features) 

# APPEARANCE
text = "Naging masaya siya pagkatapos ng lahat."
result_features = extract_appearance(text)
print(result_features) 

# KNOWLEDGE
text = "Alam niya ang mga detalye."
result_features = extract_knowledge(text)
print(result_features) 

# DESCRIPTION
text = "Sinabi nya na maaaring umulan bukas."
result_features = extract_description(text)
print(result_features) 

# SUPPOSITION
text = "Maaaring umulan bukas."
result_features = extract_supposition(text)
print(result_features) 

# SUBJECTIVATION
text = "Nakikita ko na mabuti siya."
result_features = extract_subjectivation(text)
print(result_features) 

# ATTITUDE
text = "Nagulat ako sa kanyang ginawa."
result_features = extract_attitude(text)
print(result_features) 

# COMPARATIVE
text = "Mas maganda ang bagong modelo kaysa sa lumang isa."
result_features = extract_comparative(text)
print(result_features) 

# QUANTIFIER
text = "Ang kanyang bahay ay mas malaki kaysa sa aking bahay."
result_features = extract_quantifier(text)
print(result_features) 

# QUALIFICATION
text = "Napaka-ganda ng kanyang painting."
result_features = extract_qualification(text)
print(result_features) 

# EXPLANATION
text = "Sumali siya sa club upang matuto ng bagong kasanayan."
result_features = extract_explanation(text)
print(result_features) 

