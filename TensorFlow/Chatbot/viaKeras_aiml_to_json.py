from xml.dom import minidom
from xml.dom.minidom import Node
from collections import defaultdict
from TensorFlow.Chatbot.viaKeras_utils import int2word


knowledge_files = ['wolcott_knowledge/uspto-user_training.aiml',
                   'wolcott_knowledge/uspto-academy_materials.aiml',
                   'wolcott_knowledge/uspto-mpep_toc.aiml']

# knowledge_files = ['wolcott_knowledge/test.aiml']

knowledge_corpus = {}
synonyms_corpus = defaultdict(list)

for knowledge_file in knowledge_files:
    xmldoc = minidom.parse(knowledge_file)

    category_list = xmldoc.getElementsByTagName('category')
    print("1 Category List Length : ", len(category_list))

    for category in category_list:
        pattern = ''
        template = ''
        srai_link = ''
        for x in category.childNodes:
            if x.nodeType == Node.ELEMENT_NODE:
                tag_name = x.tagName.strip()
                text_val = x.childNodes[0].wholeText.strip()
                # print("2 processing tag name '%s'" % tag_name)
                # if len(text_val) > 0:
                #     print('\t 3 text value "%s"' % text_val)

                if tag_name == 'pattern':
                    print('4 processing pattern... "%s"' % text_val)
                    if (text_val[0] in ('_', '*')) or (text_val[-1] in ('_', '*')):
                        print("\t5 reference... skip")
                        break
                    pattern = text_val
                    srai_link = text_val

                if tag_name == 'template':
                    template = x.childNodes[0].wholeText.strip()
                    print('6 processing template "%s"' % template)
                    for y in x.childNodes:
                        if y.nodeType == Node.ELEMENT_NODE:
                            # print("7 processing template... %s" % y.tagName)
                            if y.tagName == 'srai':
                                print('\t8 processing srai')
                                srai_link = y.childNodes[0].wholeText
                                print('\t\t10 srai link "%s"' % srai_link)
                                template = knowledge_corpus[srai_link.replace(" ", "_")]

        if pattern != '':
            print("11 Populating the knowledge corpus")
            print("12 pattern %s" % pattern)
            print("13 template %s" % template)
            knowledge_corpus[pattern.replace(" ", "_")] = template

            print("14 Populating the synonyms corpus")
            print("15 srai_link %s" % srai_link)
            print("16 pattern %s" % pattern)
            synonyms_corpus[srai_link.replace(" ", "_")].append(pattern)


# print(len(knowledge_corpus))
# print(len(synonyms_corpus))
# print(synonyms_corpus['MPEP LINK'])
# print(knowledge_corpus['CAN I DO A 2ND ACTION NON FINAL'])
# print(knowledge_corpus['TYPES OF PATENTS'])
#
# print(knowledge_corpus['MPEP LINK'])
# print(knowledge_corpus['LINK TO MPEP'])
# print(knowledge_corpus['MANUAL OF PATENT EXAMINING PROCEDURE LINK'])

print("knowledge and synonyms corpora are complete. start generation of the json")

import json

data = {}
data['intents'] = []
reserved_words = []
for pattern, patterns in synonyms_corpus.items():
    template = knowledge_corpus[pattern]
    # # convert int to word
    # converted_patterns = []
    # for pat in patterns:
    #     digits = [int(s) for s in pat.split() if s.isdigit()]
    #     converted_digits = []
    #     if len(digits)> 0:
    #         for digit in digits:
    #             converted_digits.append(int2word(digit))
    #         nondigits = [s for s in pat.split() if not s.isdigit()]
    #         new_pattern = " ".join(str(x) for x in nondigits) + " " + " ".join(str(x) for x in converted_digits)
    #         converted_patterns.append(new_pattern)
    # if len(converted_patterns) > 0:
    #     patterns = patterns + converted_patterns
    data['intents'].append({
        'tag': pattern,
        'patterns': patterns,
        'responses': [template]
    })

    for x in patterns:
        if len(x) > 3:
            x = x.replace('101', '')
        reserved_words.append(x)

with open('intents_uspto.json', 'w') as outfile:
    json.dump(data, outfile)

with open('reserved_words.json', 'w') as outfile:
    json.dump(reserved_words, outfile)

