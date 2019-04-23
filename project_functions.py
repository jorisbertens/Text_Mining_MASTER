# import necessary libraries
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect
import re
import string
import pandas as pd

# function for tokenizing a dictionary with the string as the value (dict{"asd":"THIS IS TOKENIZED"})
def tokenize_tw(dct):
    for i in dct:
        j = dct[i][1].lower()
        token_norm = RegexpTokenizer(r'\w+').tokenize(j)
        dct[i][1] = token_norm
    return dct


#function for removing stopwords from a dictionary with the string as the value (dict{"asd":"THIS IS TOKENIZED"})
def remove_stopwords(dct,language):
    for j in dct:
        T = dct[j][1]
        stop_words = set(stopwords.words(language))
        norm_tokens = [i for i in T if not i in stop_words]
        dct[j][1] = norm_tokens
    return dct

#function for lemmatizing  a dictionary with the string as the value (dict{"asd":"THIS IS TOKENIZED"})
def lemmatize(dct):
    wordnet = WordNetLemmatizer()
    for j in dct:
        i = dct[j][1]
        lemma = [wordnet.lemmatize(token) for token in i] 
        dct[j][1] = lemma
    return dct

#function for stemmatizing  a dictionary with the string as the value (dict{"asd":"THIS IS TOKENIZED"})
def stemmatize(dct,language):
    for j in dct:
        i = dct[j][1]
        stemmer = nltk.SnowballStemmer(language)
        stemmed = [stemmer.stem(token) for token in i]
        dct[j][1] = stemmed
    return dct

# filter the english tweets while doing the livestream
def filter_english(dictionary):
    blanco = "blanco"
    try:
        language = detect(dictionary["text"])
        if language == "en":
            dictionary["language"] = language
        else:
            dictionary = {"text": "Not English"}
    except:
        dictionary = {"text": "Not English"}
    return dictionary

# remove unecessary characters from the tweets on the leave streamed tweets
def removal_function(dictionary):
    y = dictionary["text"]

    y = re.sub(r"@[A-Z-a-z-0-9_.]+", "", y)  # remove users with@
    y = y.replace("\n", " ")  # remove enters
    y = re.sub(r"http\S+", "", y)  # removes links
    y = re.sub("\s+", " ", y)  # removes more one spaces
    y = re.sub(r"&(amp;)", "&", y)  # removes and in html format
    y = re.sub(r"[0-9]", "", y)  # remove numbers
    y = re.sub(r"(.+?)\1+", r"\1", y)  # remove repeted letters
    y = re.sub("\s+", " ", y)  # remove more one space

    dictionary["text"] = y

    return dictionary

"""
This code implements a basic, Twitter-aware tokenizer.
A tokenizer is a function that splits a string of text into words. In
Python terms, we map string and unicode objects into lists of unicode
objects.
There is not a single right way to do tokenizing. The best method
depends on the application.  This tokenizer is designed to be flexible
and this easy to adapt to new domains and tasks.  The basic logic is
this:
1. The tuple regex_strings defines a list of regular expression
   strings.
2. The regex_strings strings are put, in order, into a compiled
   regular expression object called word_re.
3. The tokenization is done by word_re.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   Tokenizer.
4. When instantiating Tokenizer objects, there is a single option:
   preserve_case.  By default, it is set to True. If it is set to
   False, then the tokenizer will downcase everything except for
   emoticons.
The __main__ method illustrates by tokenizing a few examples.
I've also included a Tokenizer method tokenize_random_tweet(). If the
twitter library is installed (http://code.google.com/p/python-twitter/)
and Twitter is cooperating, then it should tokenize a random
English-language tweet.
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution 3.0 Unported (CC BY 3.0): http://creativecommons.org/licenses/by/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

######################################################################

import re
import html.entities

######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most imporatantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
# Twitter symbols/cashtags:  # Added by awd, 20140410.
# Based upon Twitter's regex described here: <https://blog.twitter.com/2013/symbols-entities-tweets>.
cashtag_string = r"""(?:\$[a-zA-Z]{1,6}([._][a-zA-Z]{1,2})?)"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,
    # HTML tags:
    r"""(?:<[^>]+>)"""
    ,
    # URLs:
    r"""(?:http[s]?://t.co/[a-zA-Z0-9]+)"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Twitter symbols/cashtags:
    cashtag_string
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

######################################################################
# This is the core tokenizing regex:
    
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon and cashtag strings get their own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
cashtag_re = re.compile(cashtag_string, re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################

class TweetTokenizer(object):
    def __init__(self, *, preserve_case: bool=False):
        self.preserve_case = preserve_case

    def tokenize_tweet(self, tweet: str) -> list:
        """
        Argument: tweet -- any string object.
        Value: a tokenized list of strings; concatenating this list returns the original string if preserve_case=True
        """
        # Fix HTML character entitites:
        tweet = self._html2unicode(tweet)
        # Tokenize:
        matches = word_re.finditer(tweet)
        if self.preserve_case:
            return [match.group() for match in matches]
        return [self._normalize_token(match.group()) for match in matches]

    @staticmethod
    def _normalize_token(token: str) -> str:

        if emoticon_re.search(token):
            # Avoid changing emoticons like :D into :d
            return token
        if token.startswith('$') and cashtag_re.search(token):
            return token.upper()
        return token.lower()

    @staticmethod
    def _html2unicode(tweet: str) -> str:
        """
        Internal method that seeks to replace all the HTML entities in
        tweet with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(tweet))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    tweet = tweet.replace(ent, chr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(tweet))
        ents = filter((lambda x: x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                tweet = tweet.replace(ent, chr(html.entities.name2codepoint[entname]))
            except:
                pass
            tweet = tweet.replace(amp, " and ")
        return tweet

###############################################################################

if __name__ == '__main__':
    tokenizer = TweetTokenizer()
    samples = (
        u"RT @ #happyfuncoding: this is a typical Twitter tweet :-)",
        u"HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
        u"It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace.",
        u"$AAPL, http://t.co/asdFGH01, and $GOOG, &lt;https://t.co/asdFGH02&gt;, are battling it out through Google's proxy, Samsung."
        )

    for s in samples:
        print("======================================================================")
        print(s)
        tokens = tokenizer.tokenize(s)
        print("\n".join(tokens))

