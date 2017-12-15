from xml.dom import minidom
from xml.dom.minidom import Node

xmldoc = minidom.parse('uspto-user_training.aiml')

category_list = xmldoc.getElementsByTagName('category')
print("Category List Length : ", len(category_list))

knowledge_corpus = {}

for category in category_list:
    pattern = ''
    template = ''
    for x in category.childNodes:
        if x.nodeType == Node.ELEMENT_NODE:
            tag_name = x.tagName.strip()
            text_val = x.childNodes[0].wholeText.strip()
            print(tag_name)
            if len(text_val) > 0:
                print('\t' + text_val)

            if tag_name == 'pattern':
                if text_val[0] in ('_', '*'):
                    print("reference... skip")
                    break
                pattern = text_val

            if tag_name == 'template':
                template = x.childNodes[0].data
                for y in x.childNodes:
                    if y.nodeType == Node.ELEMENT_NODE:
                        print(y.tagName)
                        print(y.childNodes[0].wholeText)
    knowledge_corpus[pattern] = template

print(len(knowledge_corpus))
print(knowledge_corpus['CAN I DO A 2ND ACTION NON FINAL'])
print(knowledge_corpus['TYPES OF PATENTS'])
