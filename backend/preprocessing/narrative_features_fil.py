import spacy
import pandas as pd

nlp = spacy.load("xx_ent_wiki_sm")

def extract_mode(text):
    doc = nlp(text)
    
    mode_keywords = {
        'Possibility': ['maaari', 'pwedeng','pwede'],
        'Impossibility': ['hindi', 'walang','wala'],
        'Necessity': ['dapat', 'kailangan'],
        'Prohibition': ['bawal','hindi pwede']
    }
    
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
    
    intention_keywords = {
        'Infinitive Verbs': ['mag-aaral', 'pumunta'],

        'Modal Verbs': ['nais', 'gusto', 'hangad'],

        'Purpose Clauses': ['upang', 
        'para', 'sa layuning', 'upang mapanatili', 'para sa', 'upang makamit', 
        'upang maabot', 'sa kagustuhang', 'para makuha', 'para maging', 
        'para magtagumpay', 'para maiwasan', 'para maprotektahan', 'upang magturo', 
        'para magpaliwanag', 'para magbigay', 'para makatulong', 'upang maipakita', 
        'para mabigyan', 'para maghanda', 'upang maghanda', 'upang magplano', 
        'para magsimula', 'para magwagi', 'upang masiguro', 'para mangyari', 
        'upang mangyari', 'upang umunlad', 'upang sumulong', 'para umasenso', 
        'para mapaganda', 'upang mapabilis', 'upang maglakas-loob', 'upang magpayo', 
        'upang mag-alok', 'para makapagbigay', 'para magdulot', 'upang magdulot', 
        'para sa ikabubuti', 'upang mapaganda', 'para mapanatili', 'para maprotektahan', 
        'para magdulot', 'para magturo', 'upang magbigay', 'upang makatulong', 
        'upang masiguro', 'upang mag-ambag'],

        'Auxiliary Verbs': ['babalik', 'magiging']
    }
    
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
    
    result_keywords = {
        'Completed Actions': ['natapos', 'nagawa', 'nakuha','nagresulta', 'humantong', 'nagdulot', 
        'nagbunga', 'naghatid', 'nagbigay-daan', 'naging', 'naging sanhi', 
        'nagbunga', 'natamo', 'nakuha', 'nakamit', 'nagtagumpay', 
        'nakasama', 'nakatulong', 'naranasan', 'nakapagbigay', 'nakabuo', 
        'nakapagdulot', 'napunta', 'napatunayan', 'naabot', 'nakamtan', 
        'natupad', 'nagbunga', 'nag-dulot', ],

        'Perfect Aspect Verbs': ['nagkaroon', 'nagawa', 'umunlad', 'sumulong', 'nag-asenso', 
        'naglaho', 'naganap', 'nangyari', 'nasaksihan', 'naipakita', 
        'naipamalas', 'napagtagumpayan', 'naibalik', 'naipasa', 'naiwasan', 
        'naabot', 'nadama', 'nalaman', 'naramdaman', 
        'nakapagbago', 'napagpasyahan']
    }
    
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
    
    manner_keywords = {
        'Adverbs': ['maingat', 'maayos', 'mabilis', 'tahimik', 'malumanay', 
        'mahinahon', 'masinsin', 'magaan', 'mabagal', 'matapang', 
        'malakas', 'masigla', 'puspusan', 'paggalang', 'matiyaga', 
        'tapat', 'malasakit', 'mabait', 'matapang', 'mahinahon','masinsinan', 'mapanuri', 'masusing'],

        'Adjectives as Adverbs': ['maganda','masikap', 'pagsisikap', 'masigasig', 'masinop', 'masipag', 
        'kakayahan', 'tiwala', 'kalooban', 'pag-asa', 'pagmamahal', 'respeto', 'malakas', 'maisip', 
        'pagkilala', 'pagsasaalang-alang', 'madamdamin', 'pagsusumikap', 'pagnanais', 'kasiglahan', 'kalakasan', 
        'kasipagan', 'pangarap',]
    }
    
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
    
    aspect_keywords = {
        'Aspectual Markers': ['nag', 'naka', 'nagsa'],
        
        'Verbal Affixes': ['nag-aaral', 'natapos','patuloy','nagpatuloy', 'patuloy na nangyayari', 
        'nangyayari', 'nagpapatuloy', 'nagsimula', 'nagwakas', 
        'nagsisimula', 'natatapos', 'nagaganap', 'patuloy na', 'nangyayari pa', 
        'nagsisimula pa lang', 'nagsimula na', 'patuloy na nagaganap', 
        'nagtatapos', 'nagsimula', 'natapos na', ]
    }
    
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
    
    status_keywords = {
        'Negation Words': ['hindi', 'wala', 'huwag']
    }
    
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
    
    appearance_keywords = {
        'Transition Words': ['naging', 'pinalitan', 'nagbago','nagpakita', 
        'nagsilbing', 'nagmumungkahi', 'nagpamalas', 'nagpahayag', 
        'nagbubukas', 'nagbibigay', 'nagsasalita', 'nag-aalok', 'naglalaman', 'nagsasabi', 
        'nagsusumpa', 'nag-aangkin', 'nagpapakita ng', 'nagpapahayag ng','naglalantad ng', 
        'nagsasalita ng', 'nag-aalok ng', 'naglalaman ng','nagpapatunay ng']
    }
    
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
    
    knowledge_keywords = {
        'Knowledge Verbs': ['alam', 'nauunawaan','nalaman','nalalaman', 
        'napagtanto', 'natutunan', 'kilala', 'nalaman', 'nasusundan', 'nauunawaan', 
        'nagkakaroon','nagtuturo', 'nagbibigay ng kaalaman', 'nagtuturo ng', 'nagpapaliwanag ng', 'nagsasalita ng', 
        'nagbibigay ng impormasyon', 'nagpapahayag ng', 'nagsusuri ng', 'nagtuturo ng', 
        'nagbibigay-diin', 'nagpapahayag ng','nagbibigay-alam']
    }
    
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
    
    description_keywords = {
        'Descriptive Phrases': ['sinabi', 'nasabi', 'sinasabi','naglarawan','inilarawan', 
        'nagsalaysay', 'nagdetalye','nagpaliwanag', 'nagpapakita', 
        'nagpahayag', 'nagbibigay', 'nagpapaliwanag ', 'nagbigay', 
        'nagbibigay-diin', 'nagpapahayag', 'nagpapaliwanag']
    }
    
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
    
    supposition_keywords = {
        'Supposition Modal Verbs': ['maaaring', 'baka', 'sana','akala', 'pagpapalagay', 'kumpiyansa', 'hinuha', 
        'palagay', 'imahinasyon', 'halimbawa', 'sabi', 'tulad', 
        'sakaling', 'halimbawang', 'nagpapalagay', 'akalang', 
        'nagpapalagay', 'nag-aakala ', 'nag-iisip', 
        'nagpapalagay', 'nag-aakalang','sakali']
    }
    
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
    
    subjectivation_keywords = {
        'Perception Verbs': ['nagbigay','nagpakita', 'nagsalaysay', 'naglarawan', 
        'nagsalita', 'nagsabi', 'naikwento', 'nagsasalaysay ', 'nagbigay', 
        'nagsasabi','nakikita', 'nararamdaman', 'iniisip']
    }
    
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
    
    attitude_keywords = {
        'Emotion-related Adjectives': ['masaya', 'nalungkot', 'nagulat',
        'nagustuhan', 'hindi nagustuhan', 'pabor', 'hindi pabor', 
        'sumasang-ayon', 'hindi sumasang-ayon', 'natuwa', 'nainis', 
        'nagalit', 'nagagalit', 'malungkot', 'nakakaawa', 
        'nag-aalala', 'natuwa', 'nabahala', 'nag-alala', 'nagagalit', 
        'nag-iba ng pananaw', 'nagiging positibo', 'nagiging negatibo', 
        'nagiging neutral', 'nagiging maasahin', 'nagiging pesimista', 
        'nagiging nag-aalala', 'nagiging masaya', 'nagiging malungkot']
    }
    
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
    
    comparative_keywords = {
        'Comparative Adjectives': ['mas', 'higit','higit na' 'kaysa','mas mabuti', 
        'mas masama', 'mas mataas', 'mas mababa', 'mas mabilis', 'mas mabagal', 'mas matanda', 
        'mas bata', 'mas malaki', 'mas maliit', 'mas malakas', 'mas mahina', 'mas magaan', 'mas mabigat', 'mas maganda', 
        'mas pangit', 'mas malakas', 'mas mahina', 'mas mataas', 'mas mababa', 'mas bago', 'mas luma', 'mas makabago', 
        'mas konserbatibo']
    }
    
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
    
    quantifier_keywords = {
        'Quantifiers': ['ang lahat', 'ilan', 'wala', 'marami', 'konti',
        'karamihan', 'kaunti', 'kalahatan', 
        'iba', 'madami', 'mas marami', 'pinaka marami',
        'lahat', 'marami sa', 'konti sa', 'ang lahat', 'ilang','mas']
    }
    
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
    
    qualification_keywords = {
        'Qualifying Adjectives/Adverbs': ['napaka', 'sobra', 'talaga','mas mahusay', 
        'hindi mahusay', 'magaling', 'hindi magaling', 'kasanayan', 'hindi kasanayan', 'sanay', 'hindi sanay', 
        'mahusay', 'hindi mahusay', 'dalubhasa', 'baguhan', 'bago', 'karanasan', 'walang karanasan', 
        'mas magaling', 'hindi magaling','kwalipikado', 'hindi kwalipikado']
    }
    
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
    
    explanation_keywords = {
        'Explanation Phrases': ['dahil', 'upang', 'sapagkat','nagpaliwanag','nagpapaliwanag', 'nangatwiran', 
        'nagdiin', 'nagpakita', 'nagpapahayag', 'nagbibigay']
    }
    
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