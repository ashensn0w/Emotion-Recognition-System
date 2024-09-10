import pandas as pd
import numpy as np
import re

# Define the feature extraction functions
def extract_mode(text):
    mode_indicators = [
        'suddenly', 'abruptly', 'unexpectedly', 'instantly', 'immediately', 
        'without warning', 'bigla', 'agad', 'kaagad', 'dali-dali', 'tuloy-tuloy', 
        'dumating', 'umalis', 'nagbago', 'lumipat', 'huminto', 'nagmadali', 
        'nagsimula', 'natapos', 'pumunta', 'bumalik', 'nagtuloy', 'umiba', 
        'nagpahinga', 'tumigil', 'lumabas', 'pumasok', 'nagulat', 'nagpakita', 
        'nag-alis', 'nagpatuloy', 'sumigaw', 'nagtago', 'sumugod', 'naglayo', 
        'naghintay', 'umuwi', 'nag-usap', 'nanahimik', 'lumundag', 'bumagsak', 
        'umakyat', 'bumaba', 'umikot', 'umamin', 'naglakad', 'sumilip', 
        'bumungad', 'naglakbay', 'nagpatakbo', 'pumirma', 'umupo', 'humakbang', 
        'nagkwento', 'nag-utos', 'nagdasal'
    ]
    entities = re.findall(r'\b[A-Z][a-z]*\b', text)
    return int(any(indicator in text or entity in text for indicator in mode_indicators for entity in entities))

def extract_intention(sentence):
    intention_verbs = [
        'to', 'in order to', 'so as to', 'with the aim of', 'for the purpose of', 
        'to achieve', 'to accomplish', 'to prevent', 'to avoid', 'upang', 
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
        'upang masiguro', 'upang mag-ambag'
    ]
    words = sentence.split()
    for i, word in enumerate(words):
        if word in intention_verbs:
            if i + 1 < len(words):
                return 1
    return 0

def extract_result(sentence):
    result_indicators = [
        'resulted in', 'led to', 'caused', 'produced', 'created', 
        'brought about', 'ended up', 'achieved', 'accomplished', 
        'succeeded', 'failed', 'improved', 'worsened', 'caused damage', 
        'benefited', 'harmed', 'nagresulta', 'humantong', 'nagdulot', 
        'nagbunga', 'naghatid', 'nagbigay-daan', 'naging', 'naging sanhi', 
        'nagkaroon', 'nagbunga', 'natamo', 'nakuha', 'nakamit', 'nagtagumpay', 
        'nakasama', 'nakatulong', 'naranasan', 'nakapagbigay', 'nakabuo', 
        'nakapagdulot', 'napunta', 'napatunayan', 'naabot', 'nakamtan', 
        'natupad', 'nagbunga', 'nag-dulot', 'umunlad', 'sumulong', 'nag-asenso', 
        'naglaho', 'naganap', 'nangyari', 'nagawa', 'nasaksihan', 'naipakita', 
        'naipamalas', 'napagtagumpayan', 'naibalik', 'naipasa', 'naiwasan', 
        'naabot', 'nadama', 'nalaman', 'naramdaman', 'napag-alaman', 
        'nakapagbago', 'napagpasyahan'
    ]
    return int(any(indicator in sentence for indicator in result_indicators))

def extract_manner(sentence):
    manner_adverbs = [
        'by', 'through', 'via', 'using', 'with', 'by means of', 
        'manually', 'automatically', 'mechanically', 'digitally', 
        'electronically', 'physically', 'mentally', 'carefully', 
        'quickly', 'slowly', 'efficiently', 'inefficiently', 'thoroughly', 
        'superficially', 'maingat', 'maayos', 'mabilis', 'tahimik', 'malumanay', 
        'buong ingat', 'mahinahon', 'masinsin', 'magaan', 'mabagal', 
        'buong tapang', 'buong lakas', 'buong sigla', 'puspusan', 'may paggalang', 
        'buong tiyaga', 'may katapatan', 'buong malasakit', 'may kabaitan', 
        'may katapangan', 'may kahinahunan', 'mahinahon', 'maayos', 
        'masikap', 'may pagsisikap', 'masigasig', 'masinop', 'buong sipag', 
        'may kakayahan', 'buong tiwala', 'buong kalooban', 'may pag-asa', 
        'may pagmamahal', 'may respeto', 'buong lakas', 'buong isip', 
        'may pagkilala', 'may pagsasaalang-alang', 'buong damdamin', 
        'may pagsusumikap', 'may pagnanais', 'may kasiglahan', 'may kalakasan', 
        'may kasipagan', 'may pangarap', 'masinsinan', 'mapanuri', 'masusing', 
        'matiyaga'
    ]
    return int(any(adverb in sentence for adverb in manner_adverbs))

def extract_aspect(sentence):
    aspect_indicators = [
        'when', 'while', 'since', 'until', 'before', 'after', 'during', 
        'throughout', 'simultaneously', 'concurrently', 'previously', 
        'subsequently', 'immediately', 'soon', 'later', 'frequently', 
        'occasionally', 'rarely', 'never', 'always', 'daily', 'weekly', 
        'monthly', 'yearly', 'patuloy', 'nagpatuloy', 'patuloy na nangyayari', 
        'nangyayari', 'natapos', 'nagpapatuloy', 'nagsimula', 'nagwakas', 
        'nagsisimula', 'natatapos', 'nagaganap', 'patuloy na', 'nangyayari pa', 
        'nagsisimula pa lang', 'nagsimula na', 'patuloy na nagaganap', 
        'nagtatapos', 'nagsimula', 'natapos na', 'nagsimula', 'natapos na', 
        'nagpatuloy', 'natapos na', 'nagsimula', 'natapos na', 'nagpatuloy', 
        'natapos na', 'nagsimula', 'natapos na', 'nagpatuloy', 'nagsimula', 
        'natapos na', 'nagpatuloy', 'natapos na', 'nagsimula', 'natapos na', 
        'nagpatuloy', 'nagsimula', 'natapos na'
    ]
    return int(any(indicator in sentence for indicator in aspect_indicators))

def extract_status(text):
    status_indicators = [
        'became', 'turned', 'changed', 'shifted', 'transformed', 
        'improved', 'worsened', 'increased', 'decreased', 'started', 
        'stopped', 'began', 'ended', 'opened', 'closed', 'activated', 
        'deactivated', 'grew', 'shrunk', 'softened', 'hardened', 
        'warmed', 'cooled', 'naging', 'nagbago', 'nag-transform', 
        'napalitan', 'naging bahagi', 'naging ganap', 'naging permanente', 
        'naging pansamantala', 'naging opasyonal', 'naging pangunahing', 
        'nagbago', 'naging makabago', 'naging luma', 'naging bago', 
        'naging maganda', 'naging masama', 'naging magulo', 'naging tahimik', 
        'nagpapatuloy', 'naging matagumpay', 'naging malungkot', 
        'naging masaya', 'naging mahalaga', 'naging hindi mahalaga', 
        'naging magaan', 'naging mabigat', 'naging mabilis', 'naging mabagal', 
        'naging tahimik', 'naging maingay', 'naging mabuti', 'naging masama', 
        'nagpatuloy', 'naging abala', 'naging magulo', 'naging maayos'
    ]
    return int(any(indicator in text for indicator in status_indicators))

def extract_appearance(text):
    appearance_indicators = [
        'looks', 'appears', 'seems', 'seem', 'shows', 'exhibits', 
        'displays', 'manifests', 'presents', 'exudes', 'reveals', 
        'reflects', 'indicates', 'nagpakita', 'nagsilbing', 'nagmumungkahi', 
        'nagpapakita', 'nagpapamalas', 'nagpapahayag', 'nagbubukas', 
        'nagbibigay', 'nagsasalita', 'nag-aalok', 'naglalaman', 'nagsasabi', 
        'nagsusumpa', 'nag-aangkin', 'nagpapakita ng', 'nagpapahayag ng', 
        'naglalantad ng', 'nagsasalita ng', 'nag-aalok ng', 'naglalaman ng', 
        'nagpapatunay ng', 'nagpapakita ng', 'nagpapahayag ng', 'naglalantad ng'
    ]
    return int(any(indicator in text for indicator in appearance_indicators))

def extract_knowledge(text):
    knowledge_indicators = [
        'knows', 'understands', 'comprehends', 'realizes', 'learns', 
        'is aware of', 'recognizes', 'grasps', 'acknowledges', 
        'discovers', 'finds out', 'informs', 'educates', 'enlightens', 
        'may alam', 'nauunawaan', 'nalalaman', 'napagtanto', 'natutunan', 
        'kilala', 'nalaman', 'nasusundan', 'nauunawaan', 'nagkakaroon', 
        'nagkakaroon ng kaalaman', 'nagkaroon ng kaalaman', 'nagtuturo', 
        'nagbibigay ng kaalaman', 'nagtuturo ng', 'nagpapaliwanag ng', 
        'nagsasalita ng', 'nagbibigay ng impormasyon', 'nagpapahayag ng', 
        'nagsusuri ng', 'nagtuturo ng', 'nagbibigay-diin', 'nagpapahayag ng', 
        'nagbibigay-alam'
    ]
    return int(any(indicator in text for indicator in knowledge_indicators))

def extract_description(text):
    description_indicators = [
        'describes', 'depicts', 'portrays', 'illustrates', 'outlines', 
        'explains', 'details', 'characterizes', 'defines', 'elucidates', 
        'narates', 'naglarawan', 'nagsalaysay', 'nagbigay ng detalye', 
        'nagpapaliwanag', 'nagpapakita ng detalye', 'nagbigay ng paglalarawan', 
        'nagpapahayag ng', 'nagbibigay ng', 'nagpapakita ng', 
        'nagbibigay ng impormasyon', 'nagpapaliwanag ng', 'nagbigay ng', 
        'nagbibigay-diin', 'nagpapahayag ng', 'nagpaliwanag ng'
    ]
    return int(any(indicator in text for indicator in description_indicators))

def extract_supposition(text):
    supposition_indicators = [
        'if', 'suppose', 'presume', 'assume', 'imagine', 'hypothesize', 
        'guess', 'conjecture', 'consider', 'postulate', 'assumption', 
        'pag-aakala', 'pagpapalagay', 'kumpiyansa', 'hinuha', 
        'palagay', 'imahinasyon', 'halimbawa', 'sabi nila', 'tulad ng', 
        'kung sakali', 'kung halimbawa', 'nagpapalagay', 'nag-aakalang', 
        'nag-iisip', 'nagpapalagay na', 'nag-aakala na', 'nag-iisip na', 
        'nagpapalagay na', 'nagpapalagay', 'nagpapalagay na', 
        'nagpapalagay ng', 'nag-aakalang', 'kung halimbawa', 'kung sakali'
    ]
    return int(any(indicator in text for indicator in supposition_indicators))

def extract_subjectivation(text):
    subjectivation_indicators = [
        'I', 'we', 'our', 'my', 'his', 'her', 'their', 'your', 
        'one', 'someone', 'people', 'everybody', 'every person', 
        'the author', 'the narrator', 'the character', 'nagbigay', 
        'nagpakita', 'nag-imbento', 'nagsalaysay', 'naglarawan', 
        'nag-angkin', 'nag-utos', 'nagbigay ng', 'nagsalita', 
        'nagsabi', 'nag-kwento', 'nagsalita ng', 'nagsalaysay ng', 
        'nagbigay ng detalye', 'nagsabi ng', 'nagsalita ng', 
        'nagbigay ng paliwanag', 'nagbigay ng ideya', 'nag-utos ng', 
        'nagbigay ng opinyon', 'nagbigay ng suhestiyon', 'nagbigay ng', 
        'nagbigay ng impormasyon', 'nagbigay ng pagsusuri', 
        'nagbigay ng paliwanag', 'nagbigay ng opinyon', 'nagbigay ng suhestiyon'
    ]
    return int(any(indicator in text for indicator in subjectivation_indicators))

def extract_attitude(text):
    attitude_indicators = [
        'positive', 'negative', 'neutral', 'optimistic', 'pessimistic', 
        'hopeful', 'doubtful', 'enthusiastic', 'apathetic', 
        'supportive', 'critical', 'accepting', 'rejecting', 
        'nagustuhan', 'hindi nagustuhan', 'pabor', 'hindi pabor', 
        'sumasang-ayon', 'hindi sumasang-ayon', 'natuwa', 'nainis', 
        'nagalit', 'nagagalit', 'masaya', 'malungkot', 'nakakaawa', 
        'nag-aalala', 'natuwa', 'nabahala', 'nag-alala', 'nagagalit', 
        'nag-iba ng pananaw', 'nagiging positibo', 'nagiging negatibo', 
        'nagiging neutral', 'nagiging maasahin', 'nagiging pesimista', 
        'nagiging nag-aalala', 'nagiging masaya', 'nagiging malungkot'
    ]
    return int(any(indicator in text for indicator in attitude_indicators))

def extract_comparative(text):
    comparative_indicators = [
        'better', 'worse', 'more', 'less', 'greater', 'smaller', 
        'higher', 'lower', 'faster', 'slower', 'older', 'younger', 
        'larger', 'smaller', 'stronger', 'weaker', 'bigger', 'smaller', 
        'greater', 'lesser', 'mas mabuti', 'mas masama', 'mas mataas', 
        'mas mababa', 'mas mabilis', 'mas mabagal', 'mas matanda', 
        'mas bata', 'mas malaki', 'mas maliit', 'mas malakas', 
        'mas mahina', 'mas magaan', 'mas mabigat', 'mas maganda', 
        'mas pangit', 'mas malakas', 'mas mahina', 'mas mataas', 
        'mas mababa', 'mas bago', 'mas luma', 'mas makabago', 
        'mas konserbatibo'
    ]
    return int(any(indicator in text for indicator in comparative_indicators))

def extract_quantifier(text):
    quantifiers = [
        'all', 'some', 'none', 'several', 'few', 'many', 'much', 
        'a lot', 'little', 'more', 'most', 'every', 'each', 'every', 
        'ang lahat', 'ilan', 'wala', 'marami', 'konti', 'bawat isa', 
        'bawat isa', 'ilan', 'karamihan', 'kaunti', 'kalahatan', 
        'iba', 'madami', 'mas marami', 'pinaka marami', 'kaunti', 
        'ilan', 'wala', 'marami', 'konti', 'kaunti', 'lahat', 
        'marami sa', 'konti sa', 'ang lahat', 'ilang'
    ]
    return int(any(quantifier in text for quantifier in quantifiers))

def extract_qualification(text):
    qualifications = [
        'qualified', 'unqualified', 'proficient', 'skilled', 'expert', 
        'novice', 'inexperienced', 'talented', 'gifted', 'competent', 
        'capable', 'incapable', 'adept', 'proficient', 'unskilled', 
        'mas mahusay', 'hindi mahusay', 'magaling', 'hindi magaling', 
        'kasanayan', 'hindi kasanayan', 'sanay', 'hindi sanay', 
        'mahusay', 'hindi mahusay', 'dalubhasa', 'baguhan', 'bago', 
        'karanasan', 'walang karanasan', 'mas magaling', 'hindi magaling', 
        'kwalipikado', 'hindi kwalipikado'
    ]
    return int(any(qualification in text for qualification in qualifications))

def extract_explanation(text):
    explanation_indicators = [
        'explains', 'clarifies', 'justifies', 'accounts for', 'elucidates', 
        'demonstrates', 'illustrates', 'details', 'elaborates', 'expounds', 
        'nagpapaliwanag', 'nagbibigay-katwiran', 'nagbigay-linaw', 
        'nagbibigay-diin', 'nagpapakita', 'nagpapahayag', 'nagbibigay', 
        'nagbibigay ng paliwanag', 'nagpapaliwanag ng', 
        'nagbigay ng detalye', 'nagpapakita ng dahilan', 'nagbigay ng dahilan', 
        'nagbigay-linaw', 'nagpaliwanag ng', 'nagbibigay ng detalye'
    ]
    return int(any(indicator in text for indicator in explanation_indicators))

def extract_narrative_features(text):
    features = {
        'mode': extract_mode(text),
        'intention': extract_intention(text),
        'result': extract_result(text),
        'manner': extract_manner(text),
        'aspect': extract_aspect(text),
        'status': extract_status(text),
        'appearance': extract_appearance(text),
        'knowledge': extract_knowledge(text),
        'description': extract_description(text),
        'supposition': extract_supposition(text),
        'subjectivation': extract_subjectivation(text),
        'attitude': extract_attitude(text),
        'comparative': extract_comparative(text),
        'quantifier': extract_quantifier(text),
        'qualification': extract_qualification(text),
        'explanation': extract_explanation(text)
    }
    return features

# Create the feature matrix
def create_feature_matrix(sentences):
    feature_list = []
    for sentence in sentences:
        features = extract_narrative_features(sentence)
        feature_list.append([features[feature] for feature in sorted(features.keys())])
    return np.array(feature_list)

# Load dataset
file_path = './backend/data/lowercased_data.csv'
data = pd.read_csv(file_path)

# Extract sentences
sentences = data['sentence'].tolist()

# Print the first 10 sentences and their feature vectors
print("First 10 Sentences and their Feature Vectors:")
for i, sentence in enumerate(sentences[:10]):
    features = extract_narrative_features(sentence)
    feature_vector = [features[feature] for feature in sorted(features.keys())]
    print(f"Sentence {i+1}: {sentence}")
    print(f"Feature Vector: {feature_vector}")
    print()

# Generate and print the feature matrix for the entire dataset
feature_matrix = create_feature_matrix(sentences)
print("Feature Matrix:")
print(feature_matrix)