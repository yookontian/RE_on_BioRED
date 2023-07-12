"""def get_labels(self):
    See base class.
    return ['None', 
            'Association', 
            'Bind',
            'Comparison',
            'Conversion',
            'Cotreatment',
            'Drug_Interaction',
            'Negative_Correlation',
            'Positive_Correlation']
        
@classmethod
def get_entity_type_dict(cls):
    return {'@GeneOrGeneProductSrc$':0, 
            '@DiseaseOrPhenotypicFeatureSrc$':0,
            '@ChemicalEntitySrc$':0,
            '@GeneOrGeneProductTgt$':1,
            '@DiseaseOrPhenotypicFeatureTgt$':1,
            '@ChemicalEntityTgt$':1,}
    
def get_negative_labels(self):
    return ['None']
"""
from relations import relations


def get_labels(mode='entity'):
    """return: 
    additional_tokens: {'additional_special_tokens': ['[ENTITY]', '[/ENTITY]', '[RELATION]', '[/RELATION]', '[SRC]', '[TGT]'] + relations}
    labels: [label, label, ...]
    id2label: {0:label, 1:label, ...}
    label2id: {label:0, label:1, ...}
       """
    labels = [None]
    additional_tokens = [None]
    if mode == 'entity':
        labels = ['[PAD]', '[STOP]', '[CLS]', '[B-Gene]', '[B-Disease]', '[B-Chemical]', '[IN-Gene]', '[IN-Disease]', '[IN-Chemical]', '[OUT]']
    
    if mode == 'bert_without_ner':
        labels = ['[pad]', '[src-b]', '[src-in]', '[tgt-b]', '[tgt-in]', '[out]']
        # inputted relation without none
        additional_tokens = {'additional_special_tokens': ["[" + relation + "]" for relation in relations[1:]]}


    if mode == 'bert_w_ner':
        labels = ['[pad]', '[src]', '[ner1]', '[ner2]', '[tgt]', '[none]']
        # inputted relations of bert_w_ner doesn't include 'None'
        additional_tokens = {'additional_special_tokens': ['[ner1]', '[/ner1]', '[ner2]', '[/ner2]', ] + ["[" + relation + "]" for relation in relations[1:]]}


    if mode == 'GPT_w_ner':
        # additional_tokens = {'additional_special_tokens': ['[entity1]', '[entity2]', '[learn1]', '[learn2]', '[learn3]', '[learn4]', '[learn5]', '[learn6]'] + ["[" + relation + "]" for relation in relations]}
        additional_tokens = {'additional_special_tokens': ['[learn1]', '[learn2]', '[learn3]', '[learn4]', '[learn5]', '[learn6]']}
    id2label = {idx:label for idx, label in enumerate(labels)}

    label2id = {label:idx for idx, label in enumerate(labels)}

    return additional_tokens, labels, id2label, label2id


