from TensorFlow.Chatbot.viaKeras_constants import *


def preserve_special_words(tokens):
    for index, token in enumerate(tokens):
        combined_token = token
        longest_matched_sub_index = -1
        for sub_index, sub_token in enumerate(tokens[index+1:]):
            combined_token += sub_token
            if combined_token in USPTO_RESERVED_WORDS:
                longest_matched_sub_index = index + sub_index + 1

        if longest_matched_sub_index != -1:
            left = tokens[:index]
            right = tokens[longest_matched_sub_index+1:]
            joined = [''.join(tokens[index:longest_matched_sub_index+1])]
            tokens = left + joined + right

    return tokens


def int2word(num2conv):
    """
    from https://www.daniweb.com/programming/software-development/code/216839/number-to-word-converter-python

    convert an integer number n into a string of english words
    can be used for numbers as large as 999 vigintillion
    """
    # break the number into groups of 3 digits using slicing
    # each group representing hundred, thousand, million, billion, ...
    n3 = []
    r1 = ""
    # create numeric string
    ns = str(num2conv)
    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        # break if end of ns has been reached
        if q < -2:
            break
        else:
            if q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r

    # print n3  # test

    # break each group of 3 digits into
    # ones, tens/twenties, hundreds
    # and form a string
    nw = ""
    for i, x in enumerate(n3):
        b1 = x % 10
        b2 = (x % 100) // 10
        b3 = (x % 1000) // 100
        # print b1, b2, b3  # test
        if x == 0:
            continue  # skip
        else:
            t = thousands[i]
        if b2 == 0:
            nw = ones[b1] + t + nw
        elif b2 == 1:
            nw = tens[b1] + t + nw
        elif b2 > 1:
            nw = twenties[b2] + ones[b1] + t + nw
        if b3 > 0:
            nw = ones[b3] + "hundred " + nw
    return nw


############# globals ################
ones = ["", "one ", "two ", "three ", "four ", "five ",
        "six ", "seven ", "eight ", "nine "]
tens = ["ten ", "eleven ", "twelve ", "thirteen ", "fourteen ",
        "fifteen ", "sixteen ", "seventeen ", "eighteen ", "nineteen "]
twenties = ["", "", "twenty ", "thirty ", "forty ",
            "fifty ", "sixty ", "seventy ", "eighty ", "ninety "]
thousands = ["", "thousand ", "million ", "billion ", "trillion ",
             "quadrillion ", "quintillion ", "sextillion ", "septillion ", "octillion ",
             "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ",
             "quattuordecillion ", "quindecillion", "sexdecillion ", "septendecillion ",
             "octodecillion ", "novemdecillion ", "vigintillion "]

if __name__ == '__main__':
    # select an integer number n for testing or get it from user input
    nf = 73
    print(int2word(nf))



