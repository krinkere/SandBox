from TensorFlow.Chatbot.viaKeras_model_utils import bow, classify, response, get_context

test_sentences = ["what's \"101\" means when it comes to the world of the patents",
                  "tell me about \"101 significantly more\"",
                  "explain 102(a)(1) to me",
                  "what's the definition of public use",
                  "what's the definition of \"public use\"",
                  "i am confused about 102(a)(2)",
                  "what do they mean when they say that it is \"available to public\"?",
                  "what's 112b?",
                  "what's 112a second requirement?",
                  "what's 112a third requirement?",
                  "how do you define a prior art?",
                  "how about some misspelling in externsion of time fee",
                  "what can you tell me about trademarks",
                  "what is rule 1.105?",
                  "duty to disclose",
                  "I need a LINK TO MANUAL OF PATENT EXAMINING PROCEDURE",
                  "what can you tell me about TRADEMARK IN CLAIMS",
                  "COURT OF APPEALS FEDERAL CIRCUIT",
                  "Scott loves to ask about Rule 105",
                  "What does it mean to \"First to invent\"",
                  "tell me about prior art 103",
                  "restriction?",
                  "quayle - what is it"
                  ]

for test_sentence in test_sentences:
    print("****************************")
    print("sentence '%s'" % test_sentence)
    print("classified as %r'" % classify(test_sentence))
    print("responce given '%s'" % response(test_sentence))
    print("****************************")


'''
p = bow("is your shop open today?")
print(p)

print(classify('is your shop open today?'))
response('is your shop open today?')
response('do you take cash?')
response('what kind of mopeds do you rent?')
response('Goodbye, see you later')

# Contextualization
# We want to handle a question about renting a moped and ask if the rental is for today. That clarification question is
# a simple contextual response. If the user responds ‘today’ and the context is the rental timeframe then it’s best they
# call the rental company’s 1–800 #. No time to waste.


# If an intent wants to set context, it can do so:
# {“tag”: “rental”,
#  “patterns”: [“Can we rent a moped?”, “I’d like to rent a moped”, … ],
#  “responses”: [“Are you looking to rent today or later this week?”],
#  “context_set”: “rentalday”
#  }
# If another intent wants to be contextually linked to a context, it can do that:
# {“tag”: “today”,
#  “patterns”: [“today”],
#  “responses”: [“For rentals today please call 1–800-MYMOPED”, …],
# “context_filter”: “rentalday”
#  }
# In this way, if a user just typed ‘today’ out of the blue (no context), our ‘today’ intent won’t be processed. If
# they enter ‘today’ as a response to our clarification question (intent tag:‘rental’) then the intent is processed.
'''
'''
print(get_context())
response('we want to rent a moped')
# show context
print(get_context())
response('today')

# We defined our 'greeting' intent to clear context, as is often the case with small-talk. We add a ‘show_details’
# parameter to help us see inside.
response("Hi there!", show_details=True)
response('today')
print(classify('today'))

response("thanks, your great")
'''