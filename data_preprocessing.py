import re
import pandas as pd
from relations import relations
from transformers import AutoTokenizer, BertModel
import torch
from labels import get_labels
import random

def all_line_of_pmid(dataset, start=0):
    """Find all lines with the same pmid
    returen:
    pmid, [start_index, end_index) """
    assert type(dataset) == pd.core.frame.DataFrame
    pmid = dataset.iloc[start, 0]
    for i in range(start, len(dataset)):
        if dataset.iloc[i, 0] != pmid:
            i = i
            break
    
    return pmid, start, i

def get_original_text(dataset, start, end, lower=False):
    """Get the text of the dataset with no special tokens"""
    assert type(dataset) == pd.core.frame.DataFrame
    src_start_token = f"@{dataset.iloc[start, 1]}Src\$"
    src_end_token = f"@/{dataset.iloc[start, 1]}Src\$"
    tgt_start_token = f"@{dataset.iloc[start, 2]}Tgt\$"
    tgt_end_token = f"@/{dataset.iloc[start, 2]}Tgt\$"

    # get the string of dataset.iloc[0, 7] and remove all the strings of src_token and tgt_token
    text = re.sub(src_start_token, "", dataset.iloc[start, 7])
    text = re.sub(src_end_token, "", text)
    text = re.sub(tgt_start_token, "", text)
    text = re.sub(tgt_end_token, "", text)
    # remove the continuous space to one space and remove the space at the beginning and end of the string
    text = re.sub(r"\s+", " ", text)
    # replace the continuous dot to one dot
    text = re.sub(r"\.+", ".", text)
    # replace ". ." to "."
    text = re.sub(r"\. \.", ".", text)
    text = text.strip()
    if lower:
        text = text.lower()
    else:
        pass
    return text

def get_identifier_and_entity (dataset, start, end, lower=False):
    """return a dictionary:
    {
        "identifier": 
            "entity_type":
            "entities":
    }
    And another dictionary:
    {
        entities:
            entity_type:
            identifier:
    
    }
    
    """
    a = {}
    b = {}

    identifiers = []
    for i in range(start, end):
        src_start_token = f"@{dataset.iloc[i, 1]}Src$"
        src_end_token = f"@/{dataset.iloc[i, 1]}Src$"
        tgt_start_token = f"@{dataset.iloc[i, 2]}Tgt$"
        tgt_end_token = f"@/{dataset.iloc[i, 2]}Tgt$"
        
        # print(src_start_token, tgt_start_token)
        # add new indentifier to the list
        for n in [3, 4]:
            if dataset.iloc[i, n] not in a.keys():
                identifiers.append(dataset.iloc[i, n])
                # print(dataset.iloc[i, n])
                a [str(dataset.iloc[i, n])] = {"entity_type": dataset.iloc[i, n - 2]}

                if n == 3:
                    # output every content between the src_start and src_end token in the text train_file.iloc[n, 7]
                    pattern = re.escape(src_start_token) + r"(.*?)" + re.escape(src_end_token)
                    matches = re.findall(pattern, dataset.iloc[i, 7])

                    # # print(f"{train_file.iloc[n, 1]}:")
                    # for match in matches:
                    #     print(match)

                if n == 4:
                    pattern = re.escape(tgt_start_token) + r"(.*?)" + re.escape(tgt_end_token)
                    matches = re.findall(pattern, dataset.iloc[i, 7])

                    # for match in matches:
                    #     print(match)

                if lower:
                    matches = [item.strip().lower() for item in matches]
                else:
                    matches = [item.strip() for item in matches]
                # print("matchs: ", matches)
                matches = list(set(matches))
                a[str(dataset.iloc[i, n])]["entities"] = matches
                for item in matches:
                    if item not in b.keys():
                        b[item] = {"entity_type": dataset.iloc[i, n - 2], "identifier": str(dataset.iloc[i, n])}
                    else:
                        pass
    return a, b




# re-order the list s denpend on the first occurence of each string in the text.lower()

def reorder_list(abstract, entities, lower=False, mode="occurence"):
    """Default mode is occurence, but you can change to "length" to reorder the list by the length of the string
    mode == length, will reorder the list by the length of the string from the shortest to the longest"""
    lst = []
    if lower:
        abstract = abstract.lower()
    for key, value in entities.items():
        for entity in value["entities"]:
            # s = s + " \""+ entity + "\","
            if entity not in lst:
                lst.append(entity)
    indexed_list = [(text, abstract.index(text) if text in abstract else len(abstract)) for text in lst]
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    reordered_list = [text for text, _ in sorted_list]
    if mode == "length":
        reordered_list = sorted(reordered_list, key=lambda x: len(x))
    return reordered_list


def get_relations (dataset, relations=relations, start=None, end=None, lower=False):
    """
    return a list of tuples: (src, tgt, relation), including None relation.
    """
    relations = relations
    relation = []
    for i in range(start, end):
        src_start_token = f"@{dataset.iloc[i, 1]}Src$"
        src_end_token = f"@/{dataset.iloc[i, 1]}Src$"
        tgt_start_token = f"@{dataset.iloc[i, 2]}Tgt$"
        tgt_end_token = f"@/{dataset.iloc[i, 2]}Tgt$"
        assert dataset.iloc[i, 8] in relations
        relation.append((dataset.iloc[i, 3], dataset.iloc[i, 4], dataset.iloc[i, 8]))
    
    return relation    


"""
input of encoder: text.lower()

initialized input of decoder: [CLS]

(NOT using this, relation include None)
output of decoder: [CLS] [ENTITY] [ENTITY1]e1 ; e2 [/ENTITY1] ; [ENTITY2] e3 ; e4 [/ENTITY2] ; [/ENTITY] 
                    [RELATION] [ENTITY1] [ENTITY2] [RELATION1] ; [/ENTITY1] [/ENTITY2] [RELATION1] [/RELATION]

(Using this, relation don't include None)
output of decoder: [CLS] [ENTITY] e1 ; e2; e3 ; e4 [/ENTITY] 
                    [RELATION] [SRC] e1 [TGT] e2 [RELATION1] ; [SRC] e2 [TGT] e2 [RELATION2] [/RELATION]


{   
    pmid:
    input:
    output:
}
"""
def make_dataset(file_path, lower=False, ignore_relations=[], NER=False, NER_in=False):
    """make a dictionary for the dataset
    input is the .tsv file path
    return the hugging face dataset"""
    data_dict = {
        "pmids": [],
        "inputs": [],
        "outputs": []
    }

    pandas_data = pd.read_csv(file_path, delimiter="\t", header=None)
    relations_dict = {relations[i]: f"[RELATION{i}]" for i in range(len(relations))}

    if NER_in:
        NER_tag_dict = {'GeneOrGeneProduct':"[B-Gene]",
        'DiseaseOrPhenotypicFeature':"[B-Disease]",
        'ChemicalEntity':"[B-Chemical]",
        'in': {"[B-Gene]": '[IN-Gene]',
                "[B-Disease]": '[IN-Disease]',
                "[B-Chemical]": '[IN-Chemical]'},
        'out': '[OUT]'
        }
    else:
        NER_tag_dict = {'GeneOrGeneProduct':"[B-Gene]",
        'DiseaseOrPhenotypicFeature':"[B-Disease]",
        'ChemicalEntity':"[B-Chemical]",
        'in': {"[B-Gene]": '[IN]',
                "[B-Disease]": '[IN]',
                "[B-Chemical]": '[IN]'},
        'out': '[OUT]'
        }
    start = 0

    while start < (len(pandas_data) - 1):
        pmid, start, end = all_line_of_pmid(pandas_data, start)
        # the text didn't be .lower()
        text = get_original_text(pandas_data, start, end, lower)
        # get all entities and their identifiers
        entities, entity_to_identifier = get_identifier_and_entity(pandas_data, start, end, lower)
        # reorder the entities
        if not NER:
            reordered_entities = reorder_list(text, entities, lower)
            
            # make the output of decoder
            # entities
            output_str = ["[CLS]", "[ENTITY]"]
            identi_to_num_enti = {}
            num_identify = 1
            for i in range(len(reordered_entities)):

                output_str.append(reordered_entities[i])
                if i < len(reordered_entities) - 1:
                    output_str.append(";")

            output_str.append("[/ENTITY]")
            
            data_dict["pmids"].append(pmid)
            data_dict["inputs"].append(text)
            data_dict["outputs"].append(" ".join(output_str))
        else:
            # if NER
            reordered_entities = reorder_list(text, entities, lower, mode="length")
            splited_text = text.split(" ")
            length = len(splited_text)
            NER_tags = [NER_tag_dict["out"]] * len(splited_text)
            finded = True
            for entity in reordered_entities:
                if not finded:
                    print("error, not find")
                    break
                finded = False
                splited_entity = entity.split(" ")
                i = 0
                while i < length:
                    # print("i:", i)
                    if splited_text[i] == splited_entity[0]:
                        # print(entity)
                        if len(splited_entity) > 1:
                            for j in range(1, len(splited_entity)):
                                try:
                                    if splited_text[i + j] == splited_entity[j]:
                                        continue
                                    else:
                                        j = -1
                                        break
                                except:
                                    j = -1
                                    break
                            if j + 1 == len(splited_entity):
                                NER_tags[i] = NER_tag_dict[entity_to_identifier[entity]['entity_type']]
                                # print(NER_tag_dict[entity_to_identifier[entity]['entity_type']])
                                for k in range(i + 1, i + j + 1):
                                    NER_tags[k] = NER_tag_dict["in"][NER_tag_dict[entity_to_identifier[entity]['entity_type']]]
                                i = i + j + 1
                                # print("j:", NER_tags)
                                # print("1find: ", entity)
                                finded = True
                                continue
                            else:
                                i = i + 1
                                continue
                        else:
                            NER_tags[i] = NER_tag_dict[entity_to_identifier[entity]['entity_type']]
                            i = i + 1
                            # print("i:", NER_tags)
                            # print("2find: ", entity)
                            finded = True
                            continue
                    else:
                        # print("No:", NER_tags)
                        i = i + 1
                        continue
            assert len(splited_text) == len(NER_tags)
            data_dict["pmids"].append(pmid)
            data_dict["inputs"].append(splited_text)
            data_dict["outputs"].append(NER_tags)

           

        start = end

    
    return data_dict


# with collator
def preprocess_function(examples, tokenizer):
    inputs = examples['inputs']
    targets = examples['outputs']
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    output_id = tokenizer(targets, add_special_tokens=False, max_length=512, padding="max_length", truncation=True)
    # model_inputs["labels"] = output_id["input_ids"]
    new_dict = {}
    new_dict['input_ids'] = model_inputs.input_ids
    new_dict['attention_mask'] = model_inputs.attention_mask
    new_dict['token_type_ids'] = model_inputs.token_type_ids
    new_dict['decoder_input_ids'] = output_id.input_ids
    new_dict['labels'] = output_id.input_ids.copy()
    for i in range(len(new_dict['labels'])):
        new_dict['labels'][i] = new_dict['labels'][i][1:]
        new_dict['labels'][i].append(-100)
    new_dict['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in new_dict["labels"]]
    # encoder_input_ids, encoder_attention_mask, encoder_token_type_ids, decoder_input_ids
    return new_dict


def NER_preprocess_function(examples, tokenizer, bert=False, mode="entity"):
    max_length = 512
    begin_token = ['[B-Gene]', '[B-Disease]', '[B-Chemical]']
    input_ids = []
    labels = []
    attention_mask = []
    token_type_ids = []
    a = examples['inputs']
    b = examples['outputs']
    additional_tokens, _, id2label, label2id = get_labels(mode=mode)
    for i, word in enumerate(a):
        tokenized_word = tokenizer.encode(word, add_special_tokens=False)
        input_ids.extend(tokenized_word)
        if b[i] in begin_token:
            labels.append(label2id[b[i]])
            labels.extend([label2id[b[i]] + 3] * (len(tokenized_word) - 1))
        else:
            labels.extend([label2id[b[i]]] * len(tokenized_word))


    if len(input_ids) > (max_length - 2):
        input_ids = input_ids[:max_length - 2]
        labels = labels[:max_length - 2]

    input_ids = [2] + input_ids + [3]
    if bert:
        labels = [2] + labels + [1]
    else:
        labels = labels + [1]
    attention_mask = [1] * len(input_ids) 
    token_type_ids = [0] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    if bert:
        labels = labels + ([0] * (padding_length))
    else:
        labels = labels + ([0] * (padding_length + 1))
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        # 'decoder_input_ids': [2] + labels[:-1],
        'labels': labels
        
    }





def make_bert_re_data(file_path, lower=False, output_none=False, no_ner_input=False):
    """make a dictionary for the dataset for bert_re
    input is the .tsv file path
    return the hugging face dataset
    data_dict = {
        "pmids": [],
        "inputs": [(relation token, text)],
        "outputs": []
    }

    """
    data_dict = {
        "pmids": [],
        "input_texts": [],
        "input_relations": [],
        "outputs": []
    }

    dataset = pd.read_csv(file_path, delimiter="\t", header=None)

    # relations_dict = {relations[i]: f"[RELATION{i}]" for i in range(len(relations))}

    # tag_dict = {}
    start = 0
    if no_ner_input:
        NER_tag_dict = {'src-b':"[src-b]",
                        'src-in':"[src-in]",
                        'tgt-b':"[tgt-b]",
                        'tgt-in': "[tgt-in]",
                        "out": '[out]'
                        }


    while start < (len(dataset) - 1):
        pmid, start, end = all_line_of_pmid(dataset, start)
        # the text didn't be .lower()
        # get all entities and their identifiers
        # entities, entity_to_identifier = get_identifier_and_entity(dataset, start, end, lower)
        # reorder the entities
        # reordered_entities = reorder_list(text, entities, lower, mode="length")
        # def get_text_w_input_tokens (dataset, start, end, lower=False)
        # have a dict to store the none relations of this pmid
        # relation_temp = {}
        pmid = dataset.iloc[start, 0]

        # # ner not in the input
        if no_ner_input:
            original_text = get_original_text(dataset, start, end, lower)
            included_relation = []
            for i in range(start, end):
                if dataset.iloc[i, 8] == "None":
                    continue
                if dataset.iloc[i, 8] not in included_relation:
                    included_relation.append(dataset.iloc[i, 8])

                # text = get_original_text(dataset, i, end, lower=False)
                src_start_token = f"@{dataset.iloc[i, 1]}Src$"
                src_end_token = f"@/{dataset.iloc[i, 1]}Src$"
                tgt_start_token = f"@{dataset.iloc[i, 2]}Tgt$"
                tgt_end_token = f"@/{dataset.iloc[i, 2]}Tgt$"

                text = dataset.iloc[i, 7]
                # remove the continuous space to one space and remove the space at the beginning and end of the string
                text = re.sub(r"\s+", " ", text)
                # replace the continuous dot to one dot
                text = re.sub(r"\.+", ".", text)
                split_text = text.split(" ")

                length = len(split_text)
                ner_tags = []

                src_in = False
                tgt_in = False
                for j, word in enumerate(split_text):
                    if word == src_start_token:
                        continue
                    elif word == src_end_token:
                        src_in = False
                        continue
                    
                    elif word == tgt_start_token:
                        pass
                        continue

                    elif word == tgt_end_token:
                        tgt_in = False
                        continue

                    try:
                        if split_text[j - 1] == src_start_token:
                            src_in = True
                            ner_tags.append(NER_tag_dict['src-b'])
                        elif split_text[j - 1] == tgt_start_token:
                            tgt_in = True
                            ner_tags.append(NER_tag_dict['tgt-b'])
                        elif src_in:
                            ner_tags.append(NER_tag_dict['src-in'])
                        elif tgt_in:
                            ner_tags.append(NER_tag_dict['tgt-in'])
                        else:
                            ner_tags.append(NER_tag_dict['out'])
                    # if the first word and is not the start token
                    except:
                        ner_tags.append(NER_tag_dict['out'])
                    
                data_dict["pmids"].append(pmid)
                data_dict["input_texts"].append(original_text.split(" "))
                data_dict["input_relations"].append("[" + dataset.iloc[i, 8] + "]")
                assert len(ner_tags) == len(original_text.split(" "))
                data_dict["outputs"].append(ner_tags)

                # for the relations that the text didn't include, have a [out] tag for each word
                ner_tags = [NER_tag_dict["out"]] * len(original_text.split(" "))
                for relation in relations[1:]:
                    if relation not in included_relation:
                        data_dict["pmids"].append(pmid)
                        data_dict["input_texts"].append(original_text.split(" "))
                        data_dict["input_relations"].append("[" + relation + "]")
                        assert len(ner_tags) == len(original_text.split(" "))
                        data_dict["outputs"].append(ner_tags)
                    
        
        # ner in the input
        else:
            for i in range(start, end):
                pmid = dataset.iloc[start, 0]
                # text = get_original_text(dataset, i, end, lower=False)
                src_start_token = f"@{dataset.iloc[i, 1]}Src\$"
                src_end_token = f"@/{dataset.iloc[i, 1]}Src\$"
                tgt_start_token = f"@{dataset.iloc[i, 2]}Tgt\$"
                tgt_end_token = f"@/{dataset.iloc[i, 2]}Tgt\$"
                # relation_temp[dataset.iloc[i, 3]] = {}
                # relation_temp[dataset.iloc[i, 3]][dataset.iloc[i, 4]] = '[' + dataset.iloc[i, 8] + ']'
                """
                relation_temp:
                {
                    src:{
                        tgt: (text_w_tag, relation_token)
                    }
                }
                """
                # get the first occurence in dataset.iloc[i, 7] for src_start_token or tgt_start_token
                src_start = dataset.iloc[i, 7].find(src_start_token)
                tgt_start = dataset.iloc[i, 7].find(tgt_start_token)
                if output_none and dataset.iloc[i, 8] == "None":
                    output_line = ["[src]", "[none]", "[tgt]", "[none]"]
                    if src_start < tgt_start:
                        ner_dict = {src_start_token: "[ner1]", src_end_token: "[/ner1]", tgt_start_token: "[ner2]", tgt_end_token: "[/ner2]"}
                    else:
                        ner_dict = {tgt_start_token: "[ner2]", tgt_end_token: "[/ner2]", src_start_token: "[ner1]", src_end_token: "[/ner1]"}
                else:
                    if src_start < tgt_start:
                        ner_dict = {src_start_token: "[ner1]", src_end_token: "[/ner1]", tgt_start_token: "[ner2]", tgt_end_token: "[/ner2]"}
                        output_line = ["[src]", "[ner1]", "[tgt]", "[ner2]"]
                    else:
                        ner_dict = {tgt_start_token: "[ner2]", tgt_end_token: "[/ner2]", src_start_token: "[ner1]", src_end_token: "[/ner1]"}
                        output_line = ["[src]", "[ner2]", "[tgt]", "[ner1]"]

                # get the string of dataset.iloc[0, 7] and replace all the strings of src_start_token, src_end_token, tgt_start_token, tgt_end_token according to the ner_dict
                text_ = re.sub(src_start_token, ner_dict[src_start_token], dataset.iloc[i, 7])
                text = re.sub(src_end_token, ner_dict[src_end_token], text_)
                text_ = re.sub(tgt_start_token, ner_dict[tgt_start_token], text)
                text = re.sub(tgt_end_token, ner_dict[tgt_end_token], text_)
                # remove the continuous space to one space and remove the space at the beginning and end of the string
                text_ = re.sub(r"\s+", " ", text)
                # replace the continuous dot to one dot
                text = re.sub(r"\.+", ".", text_)
                if lower:
                    text = text.lower()
                

                none_output_line = ["[src]", "[none]", "[tgt]", "[none]"]
                for relation in relations[1:]:
                    if relation == dataset.iloc[i, 8]:
                        data_dict["pmids"].append(pmid)
                        data_dict["input_texts"].append(text)
                        data_dict["input_relations"].append("[" + relation + "]")
                        assert len(output_line) == 4
                        data_dict["outputs"].append(output_line)

                    else:
                        data_dict["pmids"].append(pmid)
                        data_dict["input_texts"].append(text)
                        data_dict["input_relations"].append("[" + relation + "]")
                        assert len(output_line) == 4
                        data_dict["outputs"].append(none_output_line)


            

        start = end


    return(data_dict)

def bert_w_ner_preprocess_function(examples, tokenizer, max_length=512, mode="bert_w_ner"):
    input_ids = []
    labels = []
    attention_mask = []
    token_type_ids = []
    batch_text = examples["input_texts"]
    batch_relation = examples["input_relations"]
    batch_output = examples["outputs"]
    additional_tokens, _, id2label, label2id = get_labels(mode=mode)
    if mode == "bert_w_ner":
        tokenized_texts = tokenizer(batch_text, add_special_tokens=False)['input_ids']

        for i, text in enumerate(tokenized_texts):
            # input_ids
            if len(text) > (max_length - 3):
                tokenized_texts[i] = text[:max_length - 3]
            
            # attention mask
            attention = [1] * (len(tokenized_texts[i]) + 3)

            # padding
            padding_length = max_length - len(tokenized_texts[i]) - 3
            

            input_ids.append([tokenizer.cls_token_id] + tokenizer.encode(batch_relation[i], add_special_tokens=False) + \
                            [tokenizer.sep_token_id] + tokenized_texts[i] + ([0] * padding_length))
            attention_mask.append(attention + ([0] * padding_length))

            # labels
            label = []
            for item in batch_output[i]:
                label = label + [label2id[item]]
            labels.append(label + ([label2id["[pad]"]] * (max_length - 4)))

            assert len(input_ids[i]) == len(attention_mask[i]) == len(labels[i]) == max_length

    elif mode == "bert_without_ner":
        for n, sentence in enumerate(batch_text):
            tokenized_sentence = tokenizer(sentence, add_special_tokens=False)['input_ids']
            label = []
            tokenized_text = []
            for tokens, original_label in zip(tokenized_sentence, batch_output[n]):
                tokenized_text.extend([token for token in tokens])
                for _ in tokens:
                    
                    label.append(label2id[original_label])
            
            assert len(tokenized_text) == len(label)
            
            # input_ids
            if len(tokenized_text) > (max_length - 3):
                tokenized_text = tokenized_text[:max_length - 3]
                label = label[:max_length - 3]
            
            # attention mask
            attention = [1] * (len(tokenized_text) + 3)

            # padding
            padding_length = max_length - len(tokenized_text) - 3
            

            input_ids.append([tokenizer.cls_token_id] + tokenizer.encode(batch_relation[n], add_special_tokens=False) + \
                            [tokenizer.sep_token_id] + tokenized_text + ([0] * padding_length))
            attention_mask.append(attention + ([0] * padding_length))
            labels.append(label + ([label2id["[pad]"]] * (max_length - len(label))))
            

        
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
        
    }



def make_GPT_re_data(file_path, lower=True, no_ner_input=False, mode='default'):
    """make a dictionary for the dataset for GPT_re
    input is the .tsv file path
    return the hugging face dataset
    data_dict = {
        "pmids": [],
        "text": [],
        "entities": [],
        "outputs": []
    }

    """
    if mode == "default":
        """make a dictionary for the dataset for GPT_re
        input is the .tsv file path
        return the hugging face dataset
        data_dict = {
            "pmids": [],
            "text": [],
            "entities": [],
            "outputs": [],
            "relations": [] (lower)
            }

        """

        data_dict = {
            "pmids": [],
            "text": [],
            "entities": [],
            "outputs": [],
            "relation": []
        }
        dropped = []
        dataset = pd.read_csv(file_path, delimiter="\t", header=None)

        # relations_dict = {relations[i]: f"[RELATION{i}]" for i in range(len(relations))}

        # tag_dict = {}
        start = 0
        while start < (len(dataset) - 1):
        # while start == 0:
            pmid, start, end = all_line_of_pmid(dataset, start)
            # the text
            text = get_original_text(dataset, start, end, lower)

            pmid = dataset.iloc[start, 0]

            # for i in range(start, start + 1):
            for i in range(start, end):
                # get entities and their identifiers of this line. the 1st item of entities is the src and the 2nd is the tgt
                entities, entity_to_identifier = get_identifier_and_entity(dataset, i, i + 1, lower)

                # 1) reorder the entities, find the first occurred entity as the entity1
                reordered_e1 = reorder_list(text, {list(entities.items())[0][0]:list(entities.items())[0][1]}, lower)
                reordered_e2 = reorder_list(text, {list(entities.items())[1][0]:list(entities.items())[1][1]}, lower)
                
                # 2) find the first occurred entity as the entity1
                

                # get the output line
                # if none
                if dataset.iloc[i, 8] == "None":
                    output_line = "the relation between the source entity 1 and the target entity 2 is None . "
                else:
                    try:
                        # if entity_to_identifier[reordered_e1[0]]['identifier'] == dataset.iloc[i, 3]:
                        if 'src' in dataset.iloc[i, 7].lower().split(reordered_e1[0])[0]:
                            # entity1 is the source
                            output_line = "the relation between source entity 1 and target entity 2 is " + dataset.iloc[i, 8] + " . "
                            # print("e1 is src")
                        elif 'tgt' in dataset.iloc[i, 7].lower().split(reordered_e1[0])[0]:
                            # entity2 is the source
                            output_line = "the relation between source entity 2 and target entity 1 is " + dataset.iloc[i, 8] + " . "
                            # print("e1 is tgt")
                    except:
                        # if there is no such key of entity_to_identifier, drop this line
                        dropped.append(i)
                        continue

                
                # for strings in reordered_e1 and reordered_e2, if there is a space before or after a dot, delete the space
                for i in range(len(reordered_e1)):
                    new_string = ".".join(reordered_e1[i].split(" . "))
                    new_string = ".".join(new_string.split(" ."))
                    new_string = ".". join (new_string.split(". "))
                    reordered_e1[i] = new_string

                for i in range(len(reordered_e2)):
                    new_string = ".".join(reordered_e2[i].split(" . "))
                    new_string = ".".join(new_string.split(" ."))
                    new_string = ".". join (new_string.split(". "))
                    reordered_e2[i] = new_string


                # get a string with the items in the reordered_e1, and split each item with ","
                entity1 = " ; ".join(reordered_e1)
                entity2 = " ; ".join(reordered_e2)
                # get the entity line
                entity_line = "entity 1 : " + entity1 + " . entity 2 : " + entity2 + " . "


                data_dict["pmids"].append(str(pmid))
                data_dict["text"].append(text.strip())
                data_dict["entities"].append(entity_line.strip())
                data_dict["outputs"].append(output_line.strip())
                data_dict["relation"].append(dataset.iloc[i, 8].lower())

            start = end

        if dropped:
            print(f"Dropped {len(dropped)} line:\n {dropped}")
        return data_dict

    if mode == "[]":
        data_dict = {
            "pmids": [],
            "text": [],
            "entities": [],
            "outputs": []
        }
        dropped = []
        dataset = pd.read_csv(file_path, delimiter="\t", header=None)

        # relations_dict = {relations[i]: f"[RELATION{i}]" for i in range(len(relations))}

        # tag_dict = {}
        start = 0
        while start < (len(dataset) - 1):
            pmid, start, end = all_line_of_pmid(dataset, start)
            # the text
            text = get_original_text(dataset, start, end, lower)

            pmid = dataset.iloc[start, 0]

            # for i in range(start, start + 1):
            for i in range(start, end):
                entity_line = ""
                output_line = ""
                # get entities and their identifiers of this line. the 1st item of entities is the src and the 2nd is the tgt
                entities, entity_to_identifier = get_identifier_and_entity(dataset, i, i + 1, lower)

                # 1) reorder the entities, find the first occurred entity as the entity1
                reordered_e1 = reorder_list(text, {list(entities.items())[0][0]:list(entities.items())[0][1]}, lower)
                reordered_e2 = reorder_list(text, {list(entities.items())[1][0]:list(entities.items())[1][1]}, lower)
                
                # 2) find the first occurred entity as the entity1


                # get a string with the items in the reordered_e1, and split each item with ","
                entity1 = " ; ".join(reordered_e1)
                entity2 = " ; ".join(reordered_e2)
                # get the entity line
                entity_line = "[entity1] : " + entity1 + " . [entity2] : " + entity2 + " . "

                # get the output line
                # if none
                if dataset.iloc[i, 8] == "None":
                    output_line = "the relation between source [entity1] and target [entity2] is [None] . "
                else:
                    try:
                        if entity_to_identifier[reordered_e1[0]]['identifier'] == dataset.iloc[i, 3]:
                            # entity1 is the source
                            output_line = "the relation between source [entity1] and target [entity2] is [" + dataset.iloc[i, 8] + "] . "
                        elif entity_to_identifier[reordered_e2[0]]['identifier'] == dataset.iloc[i, 3]:
                            # entity2 is the source
                            output_line = "the relation between source [entity2] and target [entity1] is [" + dataset.iloc[i, 8] + "] . "
                    except:
                        # if there is no such key of entity_to_identifier, drop this line
                        dropped.append(i)
                        continue
                data_dict["pmids"].append(pmid)
                data_dict["text"].append(text)
                data_dict["entities"].append(entity_line)
                data_dict["outputs"].append(output_line)
            
                

            start = end
        print(f"Dropped {len(dropped)} line:\n {dropped}")
        return(data_dict)




def GPT_w_ner_preprocess_function(examples, tokenizer, max_length=1024, mode="gpt_w_ner", infer=False):
    input_ids = []
    attention_mask = []
    labels = []
    suffix = '[learn1] [learn2] [learn3] [learn4] [learn5] [learn6]'
    batch_text = examples["text"]
    batch_entities = examples['entities']
    batch_output = examples['outputs']
    truncated = 0
    if mode == "gpt_w_ner":
        tokenized_texts = tokenizer(batch_text, add_special_tokens=False)['input_ids']

        for i, text_ids in enumerate(tokenized_texts):
            # when inferencing
            if infer:
                output_ids = tokenizer.encode(batch_entities[i] + " " + suffix, add_special_tokens=False)
                label = tokenizer.encode(batch_output[i], add_special_tokens=False)
                # input_ids
                if len(text_ids) + len(output_ids) + len(label)> max_length:
                    truncated += 1
                    # truncate the text_ids
                    tokenized_texts[i] = text_ids[:max_length - len(output_ids) - len(label)]

                input_ids.append(tokenized_texts[i] + output_ids)
                labels.append(batch_output[i])

            else:
                output_ids = tokenizer.encode(batch_entities[i] + " " + suffix + " " + batch_output[i] + " " + tokenizer.eos_token, add_special_tokens=False)
                # input_ids
                if len(text_ids) + len(output_ids) > max_length:
                    truncated += 1
                    # truncate the text_ids
                    tokenized_texts[i] = text_ids[:max_length - len(output_ids)]

                # padding
                padding_length = max_length - len(tokenized_texts[i]) - len(output_ids)
                
                if output_ids[-3] not in [42392, 42393, 42394, 42395, 42396, 42397, 42398, 42399, 42400]:
                    print (batch_output[i])

                input_ids.append(tokenized_texts[i] + output_ids + ([tokenizer.pad_token_id] * padding_length))
                attention_mask.append([1] * (max_length - padding_length) + ([0] * padding_length))
                assert len(input_ids[i]) == max_length

    elif mode == "bert_without_ner":
        pass

    if truncated > 0:
        print(f"truncated {truncated} examples")


    if infer:
        return( {
        'input_ids': input_ids,
        'labels': labels,
        })
    else:

        return( {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            
        })
    

def make_GPT_re_data_no_ner(file_path, random_seed, lower=True, ):
    """make a dictionary for the dataset for GPT_re
    input is the .tsv file path
    return the hugging face dataset
    data_dict = {
        "pmids": [],
        "text": [],
        "relation": [],
        "entities": [],
        "outputs": []
    }

    """
    random.seed(random_seed)
    data_dict = {
        "pmids": [],
        "text": [],
        "entities": [],
        "outputs": [],
        "relation": []
    }
    dropped = []
    dataset = pd.read_csv(file_path, delimiter="\t", header=None)

    # relations_dict = {relations[i]: f"[RELATION{i}]" for i in range(len(relations))}

    # tag_dict = {}
    start = 0
    while start < (len(dataset) - 1):
    # while start == 0:
        pmid, start, end = all_line_of_pmid(dataset, start)
        # the text
        text = get_original_text(dataset, start, end, lower)

        pmid = dataset.iloc[start, 0]
        included_relations = {}
        for i in range(start, end):
            if dataset.iloc[i, 8] == "None":
                continue
            else:
                included_relations[dataset.iloc[i, 8].lower()] = {
                    'entities':[],
                    'outputs':[],
                    'output_lines':[]
                }
        
        # entities_list: [(source, target), (source, target), ...)] 

        # for i in range(start, start + 1):
        for i in range(start, end):
            if dataset.iloc[i, 8] == "None":
                continue
            # get entities and their identifiers of this line. the 1st item of entities is the src and the 2nd is the tgt
            entities, entity_to_identifier = get_identifier_and_entity(dataset, i, i + 1, lower)

            # 1) reorder the entities, find the first occurred entity as the entity1
            reordered_e1 = reorder_list(text, {list(entities.items())[0][0]:list(entities.items())[0][1]}, lower, mode='length')
            reordered_e2 = reorder_list(text, {list(entities.items())[1][0]:list(entities.items())[1][1]}, lower, mode='length')
            if len(reordered_e1) == 0 or len(reordered_e2) == 0:
                dropped.append(i)
                continue
            # for strings in reordered_e1 and reordered_e2, if there is a space before or after a dot, delete the space
            for j in range(len(reordered_e1)):
                new_string = ".".join(reordered_e1[j].split(" . "))
                new_string = ".".join(new_string.split(" ."))
                new_string = ".". join (new_string.split(". "))
                reordered_e1[j] = new_string

            for j in range(len(reordered_e2)):
                new_string = ".".join(reordered_e2[j].split(" . "))
                new_string = ".".join(new_string.split(" ."))
                new_string = ".". join (new_string.split(". "))
                reordered_e2[j] = new_string

            if dataset.iloc[i, 3] == list(entities.keys())[0]:
                # reordered_e1 is the source
                output_line = f"the source is {reordered_e1[0]} and the target is {reordered_e2[0]}"
                included_relations[dataset.iloc[i, 8].lower()]['outputs'].append(output_line)
                included_relations[dataset.iloc[i, 8].lower()]['entities'].append((reordered_e1, reordered_e2))
            elif dataset.iloc[i, 3] == list(entities.keys())[1]:
                # reordered_e2 is the source
                output_line = f"the source is {reordered_e2[0]} and the target is {reordered_e1[0]}"
                included_relations[dataset.iloc[i, 8].lower()]['outputs'].append(output_line)
                included_relations[dataset.iloc[i, 8].lower()]['entities'].append((reordered_e2, reordered_e1))
            else:
                dropped.append(i)
                print("error in line: ", i)
                continue

        for r, v in included_relations.items():
            data_dict["pmids"].append(str(pmid))
            data_dict["text"].append(text.strip())
            data_dict["relation"].append(r.lower().strip())
            out_line = ""
            for line in v['outputs']:
                out_line += line.lower().strip() + " ; "
            out_line = out_line[:-3] + " . "
            data_dict["outputs"].append(out_line)
            data_dict['entities'].append(v['entities'])
        # randomly choosing a itwm that is in the relations and not in the included_relations.keys()
        # have a random index in len(relations)
        # if the index is in the included_relations.keys(), continue
        # else, add the relation to the included_relations.keys()
        random_index = random.randint(0, len(relations) - 1)
        while relations[random_index].lower() in included_relations.keys() or relations[random_index].lower() == "none":
            random_index = random.randint(0, len(relations) - 1)
        
        # add the relation to the included_relations.keys()
        data_dict["pmids"].append(str(pmid))
        data_dict["text"].append(text.strip())
        data_dict["relation"].append(relations[random_index].lower().strip())
        data_dict["outputs"].append("the source is none . ")
        data_dict['entities'].append([(['none'], ['none'])])
        start = end

    if dropped:
        print(f"Dropped {len(dropped)} line:\n {dropped}")

    return data_dict



def GPT_no_ner_preprocess_function(examples, tokenizer, max_length=1024, infer=False):
    input_ids = []
    attention_mask = []
    labels = []
    suffix = '[learn1] [learn2] [learn3] [learn4] [learn5] [learn6]'
    batch_text = examples["text"]
    batch_output = examples['outputs']
    truncated = 0
    tokenized_texts = tokenizer(batch_text, add_special_tokens=False)['input_ids']

    for i, text_ids in enumerate(tokenized_texts):
        # when inferencing
        if infer:
            output_ids = tokenizer.encode(f"for relation {examples['relation'][0]} ," + " " + suffix, add_special_tokens=False)
            label = tokenizer.encode(batch_output[i], add_special_tokens=False)
            # input_ids
            if len(text_ids) + len(output_ids) + len(label)> max_length:
                truncated += 1
                # truncate the text_ids
                tokenized_texts[i] = text_ids[:max_length - len(output_ids) - len(label)]

            input_ids.append(tokenized_texts[i] + output_ids)
            labels.append(batch_output[i])

        else:
            output_ids = tokenizer.encode(f"for relation {examples['relation'][0]} ," + " " + suffix + " " + batch_output[i] + " " + tokenizer.eos_token, add_special_tokens=False)
            # input_ids
            if len(text_ids) + len(output_ids) > max_length:
                truncated += 1
                # truncate the text_ids
                tokenized_texts[i] = text_ids[:max_length - len(output_ids)]

            # padding
            padding_length = max_length - len(tokenized_texts[i]) - len(output_ids)

            input_ids.append(tokenized_texts[i] + output_ids + ([tokenizer.pad_token_id] * padding_length))
            attention_mask.append([1] * (max_length - padding_length) + ([0] * padding_length))
            assert len(input_ids[i]) == max_length


    if truncated > 0:
        print(f"truncated {truncated} examples")


    if infer:
        return( {
        'input_ids': input_ids,
        'labels': labels,
        })
    else:

        return( {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            
        })
    