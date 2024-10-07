import spacy
import pandas as pd
import json

nlp = spacy.load("xx_ent_wiki_sm")

def extract_mode(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        mode_keywords = json.load(f)['mode']
    
    detected_modes = {
        'Possibility': False,
        'Impossibility': False,
        'Necessity': False,
        'Prohibition': False
    }

    for token in doc:
        if token.text.lower() in mode_keywords['Possibility']:
            detected_modes['Possibility'] = True
        elif token.text.lower() in mode_keywords['Impossibility']:
            detected_modes['Impossibility'] = True
        elif token.text.lower() in mode_keywords['Necessity']:
            detected_modes['Necessity'] = True
        elif token.text.lower() in mode_keywords['Prohibition']:
            detected_modes['Prohibition'] = True
        
        if token.dep_ in ['acomp', 'xcomp', 'ccomp'] and token.head.text.lower() in mode_keywords['Possibility']:
            detected_modes['Possibility'] = True
        elif token.dep_ in ['acomp', 'xcomp', 'ccomp'] and token.head.text.lower() in mode_keywords['Impossibility']:
            detected_modes['Impossibility'] = True

    return 1 if any(detected_modes.values()) else 0

def extract_intention(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        intention_keywords = json.load(f)['intention']
    
    detected_intentions = {
        'Infinitive Verbs': False,
        'Modal Verbs': False,
        'Purpose Clauses': False,
        'Auxiliary Verbs': False
    }
    
    for token in doc:
        if token.text.lower() in intention_keywords['Infinitive Verbs']:
            detected_intentions['Infinitive Verbs'] = True
        elif token.text.lower() in intention_keywords['Modal Verbs']:
            detected_intentions['Modal Verbs'] = True
        elif token.text.lower() in intention_keywords['Purpose Clauses']:
            detected_intentions['Purpose Clauses'] = True
        elif token.text.lower() in intention_keywords['Auxiliary Verbs']:
            detected_intentions['Auxiliary Verbs'] = True
        
        if token.dep_ in ['acl', 'amod', 'xcomp'] and token.head.text.lower() in intention_keywords['Modal Verbs']:
            detected_intentions['Modal Verbs'] = True
        elif token.dep_ in ['acl', 'xcomp'] and token.text.lower() in intention_keywords['Infinitive Verbs']:
            detected_intentions['Infinitive Verbs'] = True

    return 1 if any(detected_intentions.values()) else 0

def extract_result(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        result_keywords = json.load(f)['result']
        
    detected_results = {
        'Completed Actions': False,
        'Perfect Aspect Verbs': False
    }
    
    for token in doc:
        if token.text.lower() in result_keywords['Completed Actions']:
            detected_results['Completed Actions'] = True
        elif token.text.lower() in result_keywords['Perfect Aspect Verbs']:
            detected_results['Perfect Aspect Verbs'] = True
        
        if token.dep_ in ['attr', 'ccomp', 'acomp'] and token.head.text.lower() in result_keywords['Completed Actions']:
            detected_results['Completed Actions'] = True
        elif token.dep_ in ['attr', 'xcomp'] and token.text.lower() in result_keywords['Perfect Aspect Verbs']:
            detected_results['Perfect Aspect Verbs'] = True

    return 1 if any(detected_results.values()) else 0

def extract_manner(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        manner_keywords = json.load(f)['manner']
    
    detected_manners = {
        'Adverbs': False,
        'Adjectives as Adverbs': False
    }
    
    for token in doc:
        if token.text.lower() in manner_keywords['Adverbs']:
            detected_manners['Adverbs'] = True
        elif token.text.lower() in manner_keywords['Adjectives as Adverbs']:
            detected_manners['Adjectives as Adverbs'] = True
        
        if token.dep_ in ['advmod'] and token.text.lower() in manner_keywords['Adverbs']:
            detected_manners['Adverbs'] = True
        elif token.dep_ in ['amod'] and token.text.lower() in manner_keywords['Adjectives as Adverbs']:
            detected_manners['Adjectives as Adverbs'] = True

    return 1 if any(detected_manners.values()) else 0

def extract_aspect(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        aspect_keywords = json.load(f)['aspect']
    
    detected_aspects = {
        'Aspectual Markers': False,
        'Verbal Affixes': False
    }
    
    for token in doc:
        if token.text.lower() in aspect_keywords['Aspectual Markers']:
            detected_aspects['Aspectual Markers'] = True
        elif token.text.lower() in aspect_keywords['Verbal Affixes']:
            detected_aspects['Verbal Affixes'] = True
        
        if token.dep_ in ['acl', 'amod'] and any(affix in token.text.lower() for affix in aspect_keywords['Aspectual Markers']):
            detected_aspects['Aspectual Markers'] = True
        elif token.dep_ in ['acl', 'xcomp'] and any(affix in token.text.lower() for affix in aspect_keywords['Verbal Affixes']):
            detected_aspects['Verbal Affixes'] = True

    return 1 if any(detected_aspects.values()) else 0

def extract_status(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        status_keywords = json.load(f)['status']
    
    detected_status = {
        'Negation Words': False
    }
    
    for token in doc:
        if token.text.lower() in status_keywords['Negation Words']:
            detected_status['Negation Words'] = True
        
        if token.dep_ in ['neg'] and token.text.lower() in status_keywords['Negation Words']:
            detected_status['Negation Words'] = True

    return 1 if any(detected_status.values()) else 0

def extract_appearance(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        appearance_keywords = json.load(f)['appearance']
    
    detected_appearance = {
        'Transition Words': False
    }
    
    for token in doc:
        if token.text.lower() in appearance_keywords['Transition Words']:
            detected_appearance['Transition Words'] = True
        
        if token.dep_ in ['ccomp', 'acl'] and token.text.lower() in appearance_keywords['Transition Words']:
            detected_appearance['Transition Words'] = True

    return 1 if any(detected_appearance.values()) else 0

def extract_knowledge(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        knowledge_keywords = json.load(f)['knowledge']
    
    detected_knowledge = {
        'Knowledge Verbs': False
    }
    
    for token in doc:
        if token.text.lower() in knowledge_keywords['Knowledge Verbs']:
            detected_knowledge['Knowledge Verbs'] = True
        
        if token.dep_ in ['ccomp', 'acl'] and token.text.lower() in knowledge_keywords['Knowledge Verbs']:
            detected_knowledge['Knowledge Verbs'] = True

    return 1 if any(detected_knowledge.values()) else 0

def extract_description(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        description_keywords = json.load(f)['description']
    
    detected_descriptions = {
        'Descriptive Phrases': False
    }
    
    for token in doc:
        if token.text.lower() in description_keywords['Descriptive Phrases']:
            detected_descriptions['Descriptive Phrases'] = True
        
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in description_keywords['Descriptive Phrases']:
            detected_descriptions['Descriptive Phrases'] = True

    return 1 if any(detected_descriptions.values()) else 0

def extract_supposition(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        supposition_keywords = json.load(f)['supposition']
    
    detected_suppositions = {
        'Supposition Modal Verbs': False
    }
    
    for token in doc:
        if token.text.lower() in supposition_keywords['Supposition Modal Verbs']:
            detected_suppositions['Supposition Modal Verbs'] = True
        
        if token.dep_ in ['xcomp', 'acl'] and token.text.lower() in supposition_keywords['Supposition Modal Verbs']:
            detected_suppositions['Supposition Modal Verbs'] = True

    return 1 if any(detected_suppositions.values()) else 0

def extract_subjectivation(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        subjectivation_keywords = json.load(f)['subjectivation']

    detected_subjectivation = {
        'Perception Verbs': False
    }
    
    for token in doc:
        if token.text.lower() in subjectivation_keywords['Perception Verbs']:
            detected_subjectivation['Perception Verbs'] = True
        
        if token.dep_ in ['ccomp', 'xcomp'] and token.text.lower() in subjectivation_keywords['Perception Verbs']:
            detected_subjectivation['Perception Verbs'] = True

    return 1 if any(detected_subjectivation.values()) else 0

def extract_attitude(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        attitude_keywords = json.load(f)['attitude']
    
    detected_attitudes = {
        'Emotion-related Adjectives': False
    }
    
    for token in doc:
        if token.text.lower() in attitude_keywords['Emotion-related Adjectives']:
            detected_attitudes['Emotion-related Adjectives'] = True
        
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in attitude_keywords['Emotion-related Adjectives']:
            detected_attitudes['Emotion-related Adjectives'] = True

    return 1 if any(detected_attitudes.values()) else 0

def extract_comparative(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        comparative_keywords = json.load(f)['comparative']
    
    detected_comparative = {
        'Comparative Adjectives': False
    }
    
    for token in doc:
        if token.text.lower() in comparative_keywords['Comparative Adjectives']:
            detected_comparative['Comparative Adjectives'] = True
        
        if token.dep_ in ['amod', 'acomp'] and token.text.lower() in comparative_keywords['Comparative Adjectives']:
            detected_comparative['Comparative Adjectives'] = True

    return 1 if any(detected_comparative.values()) else 0

def extract_quantifier(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        quantifier_keywords = json.load(f)['quantifier']
    
    detected_quantifiers = {
        'Quantifiers': False
    }
    
    for token in doc:
        if token.text.lower() in quantifier_keywords['Quantifiers']:
            detected_quantifiers['Quantifiers'] = True
        
        if token.dep_ in ['amod', 'nummod'] and token.text.lower() in quantifier_keywords['Quantifiers']:
            detected_quantifiers['Quantifiers'] = True

    return 1 if any(detected_quantifiers.values()) else 0

def extract_qualification(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        qualification_keywords = json.load(f)['qualification']
    
    detected_qualifications = {
        'Qualifying Adjectives/Adverbs': False
    }
    
    for token in doc:
        if token.text.lower() in qualification_keywords['Qualifying Adjectives/Adverbs']:
            detected_qualifications['Qualifying Adjectives/Adverbs'] = True
        
        if token.dep_ in ['amod', 'advmod'] and token.text.lower() in qualification_keywords['Qualifying Adjectives/Adverbs']:
            detected_qualifications['Qualifying Adjectives/Adverbs'] = True

    return 1 if any(detected_qualifications.values()) else 0

def extract_explanation(text):
    doc = nlp(text)
    
    with open('./backend/data/filipino_keywords.json', 'r', encoding='utf-8') as f:
        explanation_keywords = json.load(f)['explanation']
    
    detected_explanations = {
        'Explanation Phrases': False
    }
    
    for token in doc:
        if token.text.lower() in explanation_keywords['Explanation Phrases']:
            detected_explanations['Explanation Phrases'] = True
        
        if token.dep_ in ['mark', 'ccomp'] and token.text.lower() in explanation_keywords['Explanation Phrases']:
            detected_explanations['Explanation Phrases'] = True

    return 1 if any(detected_explanations.values()) else 0

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

def extract_fil_features_from_dataframe(df):
    df['features'] = df['sentence'].apply(create_feature_vector)

    features_df = pd.DataFrame(df['features'].tolist(), columns=[
        'mode', 'intention', 'result', 'manner',
        'aspect', 'status', 'appearance', 'knowledge',
        'description', 'supposition', 'subjectivation', 'attitude',
        'comparative', 'quantifier', 'qualification', 'explanation'
    ])

    return features_df