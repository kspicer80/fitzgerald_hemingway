from nltk import word_tokenize
from nltk.util import ngrams
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import spacy
nlp = spacy.load('en_core_web_lg')

def count_metaphors_nltk(text):
    sents = nltk.sent_tokenize(text)
    metaphor_count = 0
    for sent in sents:
        words = nltk.word_tokenize(sent)
        for word in words:
            synsets = wordnet.synsets(word)
            for syn in synsets:
                if syn.pos() == 's':
                    metaphor_count += 1
                    break
    return metaphor_count

def extract_metaphors_similes(text):
    # load the spaCy model
    nlp = spacy.load("en_core_web_lg")
    # process the text
    doc = nlp(text)
    # create a list to store the metaphors and similes
    metaphor_simile_count = 0
    for token in doc:
        # check if the token is an adjective or adverb
        if token.pos_ in ["ADJ", "ADV"]:
            # check if the token is used in a comparison (using the "compound" dependency label)
            if "compound" in [child.dep_ for child in token.children]:
                # add the token to the list of metaphors and similes
                metaphor_simile_count += 1
    return metaphor_simile_count

test_paragraph = "The classroom was a zoo. The alligator's teeth are white daggers. She was such a peacock, strutting around with her colorful new hat. My teacher is a dragon ready to scold anyone he looks at. Mary's eyes were fireflies. The computers at school are old dinosaurs. He is a night owl. Maria is a chicken. The wind was a howling wolf. The ballerina was a swan, gliding across the stage. Jamal was a pig at dinner. The kids were monkeys on the jungle gym. My dad is a road hog. The stormy ocean was a raging bull. The thunder was a mighty lion. In this summer heat, the kids were just a bunch of lazy dogs. The snow was a white blanket over the sleepy town. He is a shining star on that stage. Her fingers were icicles after playing outside. Her long hair was a flowing golden river. The children were flowers grown in concrete gardens. The falling snowflakes are dancers. The calm lake was a mirror. You are my sunshine. The moon was a white balloon floating over the city. The road ahead was a ribbon stretching across the desert. The park was a lake after the rain. The sun is a golden ball. The clouds are balls of cotton. The lightning was fireworks in the sky. That lawn was a perfect green carpet after getting mowed this morning. The stars are sparkling diamonds. Ben's temper was a volcano ready to explode. Those best friends are two peas in a pod."
sentence_1_count = count_metaphors_nltk(test_sentences[1])
print(sentence_1_count)