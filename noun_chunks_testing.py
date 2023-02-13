import spacy
nlp = spacy.load('en_core_web_lg')

test = "The cold water poured over the silent cup."

doc_text = nlp(test)

direct_object_counts = 0
for chunk in doc_text.noun_chunks:
    if chunk.root.dep_ == 'dobj':
        direct_object_counts += 1

print(direct_object_counts)
#doc_objs = [doc.noun_chunks for doc in doc_text]
#dependency_counts = [len([obj for obj in doc_obj if obj.root.dep_ == 'dobj']) for doc_obj in doc_objs]
#print(dependency_counts)
