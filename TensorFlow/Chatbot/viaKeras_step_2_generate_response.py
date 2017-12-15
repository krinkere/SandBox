from TensorFlow.Chatbot.viaKeras_utils import bow, classify, response, get_context

print(classify('what we need to do for 101?'))
print(classify('what we need to do for 101 that deals with significantly more for patents?'))
print(classify('what about 102(a)(1)?'))
print(classify('public use? what the hell does that mean?'))
print(classify('now, i am confused. what"s 102(a)(2) then?'))
print(classify('what do they mean when they say it is available to the public?'))


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