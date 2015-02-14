# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Chuk Orakwue (chukwuchebem.orakwue@gmail.com).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logbook
from multiprocessing import Pool
from nltk import clean_html
from bs4 import BeautifulSoup
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

logger = logbook.Logger('8K CORPUS BUILDER')

# Adjust below this line
#------------------------------

# Timespan (in business days) from earnings release when we consider return.
# This very useful in deciding the accuracy of our classifier.
# It also indirectly weighs our algo as short/mid/long-term.
# In this case, we do 10 Bdays i.e. 2 weeks. 

TIMESPAN = 10

# Return boundary, for 3-class classifier
# neg iff {-inf, -X}, neut iff [-X, X], pos iff (X, inf)
LIMIT = 0.05

#--------------------------------
# Adjust above this line 

FILENAME_PATTERN = re.compile('(\w+)-(.+)-8-K\.txt', re.IGNORECASE)

CORPORA_PATH = os.path.join(os.path.expanduser("~"), 'sec_edgar', 'corpra', str(TIMESPAN))
FORM_DIR = os.path.join(os.path.expanduser("~"), 'sec_edgar', 'data')

EQUITIES = {}

def filelistings():
    """Returns dict of files in FORM_DIR"""
    from collections import defaultdict
    from pandas import Timestamp
    filedict = defaultdict(list)
    for root, dirs, files in os.walk(FORM_DIR):
        for name in files:
            if '8-K.txt' in name:
                match = re.match(FILENAME_PATTERN, name)
                if match:
                    symbol, datetime = match.groups()
                    timestamp = Timestamp(datetime)
                    filepath = os.path.join(root, name)
                    filedict[symbol].append((symbol, timestamp, filepath))
    return filedict
    
def parser_dir(files):
    # We start by traversing our FORM-DIR downloaded with sec-edgar.py
    # Binary classification as neg/pos/neut is based on return on X days
    # post announcement. Xantos trading platform is required.
    # Else, use DataReader in pandas library to get historical returns.
    for symbol, timestamp, filepath in files:
        parsed_filename = get_parsed_filename(symbol, timestamp)

        if is_parsed(parsed_filename):
            continue
        
        logger.info('Parsing {}'.format(filepath))
        er = extract_er(read_file(filepath))

        if er and 'quarter' in er:
            clean_er = clean_text(killgremlins(er))
            er_class = classify(symbol=symbol, timestamp=timestamp)
            corpra_path = os.path.join(CORPORA_PATH, er_class, parsed_filename)
            mkdir(os.path.join(CORPORA_PATH,er_class))
            write_file(corpra_path, clean_er)
    
def classify(symbol, timestamp, timespan=TIMESPAN, 
             num_class=3, adjust_delay=-4, limit=LIMIT):
    """
    Classifies press release information as positive or negative
    with respect to improve on stock close price `timespan`-days post release.

    Params:
    -------

    symbol : string
        Symbol/ticker

    timestamp : datetime-like
        Timestamp of event
        
    adjust_delay: int, Default: -4
        # of business days to add/remove from timestamp to account for potential
        delay in filling to SEC. Default of 4 is for Item 9.
        See http://www.sec.gov/answers/form8k.htm
        
    num_class: int, Default: 3
        # of class for classfier
        3-class: ('neg', 'neutral', or 'pos'), 'neutral' = lack of sentiment.
        2-class: ('neg', 'pos')
        
    timespan : int, Default: 10
        # of business days from timestamp which we classify.

    limit : float, Default: 3% (0.03)
        Limit at which event is classified as 'neg', 'pos' or 'neutral'

    Returns:
    --------
        string
            3-class: ('neg', 'neutral', or 'pos')
            2-class: ('neg', 'pos')

    """
    global EQUITIES
    from xantos import Equity
    from pandas import DatetimeIndex
    from pandas.tseries.offsets import BDay
    
    assert num_class in (2, 3)
    
    adjusted_timestamp = timestamp + BDay(adjust_delay)
    dt_index = DatetimeIndex(start=adjusted_timestamp, periods=timespan, freq='B')
    
    if symbol not in EQUITIES:
        EQUITIES[symbol] = Equity(symbol)

    close_prices = EQUITIES[symbol].historical.ix[dt_index]['Close']
    returns = close_prices.pct_change()
    cum_returns = ((1 + returns).cumprod() - 1)[-1]

    if num_class == 3:
        sentiment = 'neg' if cum_returns < -1*limit else 'pos' if cum_returns > limit else 'neut'
    elif num_class == 2:
        sentiment = 'neg' if cum_returns < limit else 'pos'
    
#    logger.info('Classification: {sentiment}\n\n{prices}'.format(sentiment=sentiment,
#                prices=close_prices))
    
    return sentiment

def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except:
            pass
            
def write_file(filepath, string_buffer):
    if string_buffer:
        with open(filepath, 'w+') as f:
            f.writelines(string_buffer)

def read_file(filepath):
    """Lazy file read"""
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            yield line

def er_generator(lines):
    """Yields First Exhibit 99. Earnings is 99.1 usually"""
    start_marker_found, return_line = False, False
    pattern = re.compile('.+(announces|reports|accounced|reported).+(quarter).+',
                         flags=re.I|re.M)

    for line in lines:
        if not start_marker_found:
            if r'<TYPE>EX-99' in line:
                start_marker_found = True 
            elif re.search(pattern, line):
                return_line = True
        if start_marker_found and r'<TEXT>' in line:
            return_line = True
        if return_line:
            yield line
        if r'</TEXT>' in line and start_marker_found:
            break
    
def replace_with_newlines(element):
    import types
    text = ''
    for elem in element.recursiveChildGenerator():
        if isinstance(elem, types.StringTypes):
            text += elem.strip()
        elif elem.name == 'br':
            text += '\n'
    return text

#http://stackoverflow.com/questions/12357261/handling-non-standard-american-english-characters-and-symbols-in-a-csv-using-py
cp1252 = {

    u"\x80": u"\u20AC",    #            e282ac
    u"\x81": u"\uFFFD",    #    `   ?    efbfbd
    u"\x82": u"\u201A",    #            e2809a
    u"\x83": u"\u0192",    #    à   à   c692
    u"\x84": u"\u201E",    #    Ġ   Ġ   e2809e
    u"\x85": u"\u2026",    #    Š   Š   e280a6
    u"\x86": u"\u2020",    #    Ơ   Ơ   e280a0
    u"\x87": u"\u2021",    #    Ǡ   Ǡ   e280a1
    u"\x88": u"\u02C6",    #    Ƞ   Ƞ   cb86
    u"\x89": u"\u2030",    #    ɠ   ɠ   e280b0
    u"\x8a": u"\u0160",    #    ʠ   ʠ   c5a0
    u"\x8b": u"\u2039",    #    ˠ   ˠ   e280b9
    u"\x8c": u"\u0152",    #    ̠   ̠   c592
    u"\x8d": u"\uFFFD",    #    ͠   ?    efbfbd
    u"\x8e": u"\u017D",    #    Π   Π   c5bd
    u"\x8f": u"\uFFFD",    #    Ϡ   ?    efbfbd
    u"\x90": u"\uFFFD",    #    Р   ?    efbfbd
    u"\x91": u"\u2018",    #    Ѡ   Ѡ   e28098
    u"\x92": u"\u2019",    #    Ҡ   Ҡ   e28099
    u"\x93": u"\u201C",    #    Ӡ   Ӡ   e2809c
    u"\x94": u"\u201D",    #    Ԡ   Ԡ   e2809d
    u"\x95": u"\u2022",    #    ՠ   ՠ   e280a2
    u"\x96": u"\u2013",    #    ֠   ֠   e28093
    u"\x97": u"\u2014",    #    נ   נ   e28094
    u"\x98": u"\u02DC",    #    ؠ   ؠ   cb9c
    u"\x99": u"\u2122",    #    ٠   ٠   e284a2
    u"\x9a": u"\u0161",    #    ڠ   ڠ   c5a1
    u"\x9b": u"\u203A",    #    ۠   ۠   e280ba
    u"\x9c": u"\u0153",    #    ܠ   ܠ   c593
    u"\x9d": u"\uFFFD",    #    ݠ   ?    efbfbd
    u"\x9e": u"\u017E",    #    ޠ   ޠ   c5be
    u"\x9f": u"\u0178",    #    ߠ   ߠ   c5b8
    u"\xa0": u"\u00A0",    #             c2a0
    u"\xa1": u"\u00A1",    #    `   `   c2a1
    u"\xa2": u"\u00A2",    #            c2a2
    u"\xa3": u"\u00A3",    #    à   à   c2a3
    u"\xa4": u"\u00A4",    #    Ġ   Ġ   c2a4
    u"\xa5": u"\u00A5",    #    Š   Š   c2a5
    u"\xa6": u"\u00A6",    #    Ơ   Ơ   c2a6
    u"\xa7": u"\u00A7",    #    Ǡ   Ǡ   c2a7
    u"\xa8": u"\u00A8",    #    Ƞ   Ƞ   c2a8
    u"\xa9": u"\u00A9",    #    ɠ   ɠ   c2a9
    u"\xaa": u"\u00AA",    #    ʠ   ʠ   c2aa
    u"\xab": u"\u00AB",    #    ˠ   ˠ   c2ab
    u"\xac": u"\u00AC",    #    ̠   ̠   c2ac
    u"\xad": u"\u00AD",    #    ͠   ͠   c2ad
    u"\xae": u"\u00AE",    #    Π   Π   c2ae
    u"\xaf": u"\u00AF",    #    Ϡ   Ϡ   c2af
    u"\xb0": u"\u00B0",    #    Р   Р   c2b0
    u"\xb1": u"\u00B1",    #    Ѡ   Ѡ   c2b1
    u"\xb2": u"\u00B2",    #    Ҡ   Ҡ   c2b2
    u"\xb3": u"\u00B3",    #    Ӡ   Ӡ   c2b3
    u"\xb4": u"\u00B4",    #    Ԡ   Ԡ   c2b4
    u"\xb5": u"\u00B5",    #    ՠ   ՠ   c2b5
    u"\xb6": u"\u00B6",    #    ֠   ֠   c2b6
    u"\xb7": u"\u00B7",    #    נ   נ   c2b7
    u"\xb8": u"\u00B8",    #    ؠ   ؠ   c2b8
    u"\xb9": u"\u00B9",    #    ٠   ٠   c2b9
    u"\xba": u"\u00BA",    #    ڠ   ڠ   c2ba
    u"\xbb": u"\u00BB",    #    ۠   ۠   c2bb
    u"\xbc": u"\u00BC",    #    ܠ   ܠ   c2bc
    u"\xbd": u"\u00BD",    #    ݠ   ݠ   c2bd
    u"\xbe": u"\u00BE",    #    ޠ   ޠ   c2be
    u"\xbf": u"\u00BF",    #    ߠ   ߠ   c2bf
    u"\xc0": u"\u00C0",    #            c380
    u"\xc1": u"\u00C1",    #    `   `   c381
    u"\xc2": u"\u00C2",    #            c382
    u"\xc3": u"\u00C3",    #    à   à   c383
    u"\xc4": u"\u00C4",    #    Ġ   Ġ   c384
    u"\xc5": u"\u00C5",    #    Š   Š   c385
    u"\xc6": u"\u00C6",    #    Ơ   Ơ   c386
    u"\xc7": u"\u00C7",    #    Ǡ   Ǡ   c387
    u"\xc8": u"\u00C8",    #    Ƞ   Ƞ   c388
    u"\xc9": u"\u00C9",    #    ɠ   ɠ   c389
    u"\xca": u"\u00CA",    #    ʠ   ʠ   c38a
    u"\xcb": u"\u00CB",    #    ˠ   ˠ   c38b
    u"\xcc": u"\u00CC",    #    ̠   ̠   c38c
    u"\xcd": u"\u00CD",    #    ͠   ͠   c38d
    u"\xce": u"\u00CE",    #    Π   Π   c38e
    u"\xcf": u"\u00CF",    #    Ϡ   Ϡ   c38f
    u"\xd0": u"\u00D0",    #    Р   Р   c390
    u"\xd1": u"\u00D1",    #    Ѡ   Ѡ   c391
    u"\xd2": u"\u00D2",    #    Ҡ   Ҡ   c392
    u"\xd3": u"\u00D3",    #    Ӡ   Ӡ   c393
    u"\xd4": u"\u00D4",    #    Ԡ   Ԡ   c394
    u"\xd5": u"\u00D5",    #    ՠ   ՠ   c395
    u"\xd6": u"\u00D6",    #    ֠   ֠   c396
    u"\xd7": u"\u00D7",    #    נ   נ   c397
    u"\xd8": u"\u00D8",    #    ؠ   ؠ   c398
    u"\xd9": u"\u00D9",    #    ٠   ٠   c399
    u"\xda": u"\u00DA",    #    ڠ   ڠ   c39a
    u"\xdb": u"\u00DB",    #    ۠   ۠   c39b
    u"\xdc": u"\u00DC",    #    ܠ   ܠ   c39c
    u"\xdd": u"\u00DD",    #    ݠ   ݠ   c39d
    u"\xde": u"\u00DE",    #    ޠ   ޠ   c39e
    u"\xdf": u"\u00DF",    #    ߠ   ߠ   c39f
    u"\xe0": u"\u00E0",    #    ࠠ  ࠠ  c3a0
    u"\xe1": u"\u00E1",    #    ᠠ  ᠠ  c3a1
    u"\xe2": u"\u00E2",    #    ⠠  ⠠  c3a2
    u"\xe3": u"\u00E3",    #    㠠  㠠  c3a3
    u"\xe4": u"\u00E4",    #    䠠  䠠  c3a4
    u"\xe5": u"\u00E5",    #    堠  堠  c3a5
    u"\xe6": u"\u00E6",    #    栠  栠  c3a6
    u"\xe7": u"\u00E7",    #    砠  砠  c3a7
    u"\xe8": u"\u00E8",    #    蠠  蠠  c3a8
    u"\xe9": u"\u00E9",    #    頠  頠  c3a9
    u"\xea": u"\u00EA",    #    ꠠ  ꠠ  c3aa
    u"\xeb": u"\u00EB",    #    렠  렠  c3ab
    u"\xec": u"\u00EC",    #    젠  젠  c3ac
    u"\xed": u"\u00ED",    #    ��  ��  c3ad
    u"\xee": u"\u00EE",    #        c3ae
    u"\xef": u"\u00EF",    #        c3af
    u"\xf0": u"\u00F0",    #    𠠠 𠠠 c3b0
    u"\xf1": u"\u00F1",    #    񠠠 񠠠 c3b1
    u"\xf2": u"\u00F2",    #    򠠠 򠠠 c3b2
    u"\xf3": u"\u00F3",    #    󠠠 󠠠 c3b3
    u"\xf4": u"\u00F4",    #    ���� ���� c3b4
    u"\xf5": u"\u00F5",    #    ���� ���� c3b5
    u"\xf6": u"\u00F6",    #    ���� ���� c3b6
    u"\xf7": u"\u00F7",    #    ���� ���� c3b7
    u"\xf8": u"\u00F8",    #    𠠠 𠠠 c3b8
    u"\xf9": u"\u00F9",    #    񠠠 񠠠 c3b9
    u"\xfa": u"\u00FA",    #    򠠠 򠠠 c3ba
    u"\xfb": u"\u00FB",    #    󠠠 󠠠 c3bb
    u"\xfc": u"\u00FC",    #    ���� ���� c3bc
    u"\xfd": u"\u00FD",    #    ���� ���� c3bd
    u"\xfe": u"\u00FE",    #    ���� ���� c3be
    u"\xff": u"\u00FF",    #    ���� ���� c3bf

}

def killgremlins(text):
    # map cp1252 gremlins to real unicode characters
    if re.search(u"[\x80-\xff]", text):
        def fixup(m):
            s = m.group(0)
            return cp1252.get(s, s)
        if isinstance(text, type("")):
            # make sure we have a unicode string
            text = unicode(text, "iso-8859-1")
        text = re.sub(u"[\x80-\xff]", fixup, text)
    return text
       
def extract_er(lines):
    """Extract the Earning Press Release from filing.
    Removes financial tables >some<
    """
    
    try:
        string_buffer = StringIO(''.join(er_generator(lines)))
        soup = BeautifulSoup(string_buffer)
    
        # Remove all financial tables
        # Do only if mostly numbers.
        for s in soup.find_all('table'):
            s.extract()
        
        paragraphs = soup.find('body').find_all('p')
        if not paragraphs:
            paragraphs = soup.find('body').find_all('div')
        if len(paragraphs) > 1:
            body = clean_html('\n\n'.join((s.text.replace('\n', ' ') for s in paragraphs)))
        else:
            body = clean_html(soup.find('body').text)
            
        return body
        
    except AttributeError:
        return ''

def is_parsed(parsed_filename):
    """Return True if parsed (classified) filing already exists, false otherwise"""
    return os.path.exists(os.path.join(CORPORA_PATH, 'neg', parsed_filename)) or \
        os.path.exists(os.path.join(CORPORA_PATH, 'pos', parsed_filename))

def get_parsed_filename(symbol, timestamp):
    return '{symbol}-{date}.txt'.format(
        symbol=symbol,date=timestamp.strftime('%Y-%m-%d'))
        
def clean_text_helper(text):
    """Helper to remove lines with below patterns"""
    # todo: leverage multiprocessing    
    flags = re.I|re.M|re.S

    PATTERNS = {
        'ACCOUNTING': re.compile('^.+accordance with.+(generally accepted|GAAP).+$',flags=flags),
        'FORWARD-LOOKING': re.compile('^\s*.*forward-looking.+statement.*$',flags=flags),
        'CAUTIONARY-STATEMENT': re.compile('^\s*.*cautionary.+statement.*$',flags=flags),
        'CONFERENCE CALL': re.compile('^\s*.*conference call.*$',flags=flags),
        'LIVE WEBCAST': re.compile('^.+live webcast.+$',flags=flags),
        'COMPANY INFO1': re.compile(r'^.+ is.+(a|an|the).+under.+the.+symbol [A-Z]+.$', flags=flags),
        'COMPANY INFO2': re.compile(r'^.+founded.+is (a|an|the)', flags=flags),
        'COMPANY INFO3': re.compile(r'^Founded.+is', flags=flags),
        'COMPANY INFO4': re.compile(r'.+(additional|more).+information.+', flags=flags),
        'COMPANY INFO5': re.compile(r'^.+ is (a|an|the).+company.+', flags=flags),
# these are super-expensive!
#        'COMPANY INFO6': re.compile(r'^.+\D*\w*\D*\s*\w+\D*.+is (a|an|the).+', flags=flags),
#        'COMPANY CONTACT1': re.compile(r'.+[^@]+@[^@]+\.[^@]+$', flags=flags), 
        'COMPANY CONTACT2': re.compile(r'.+\d{1}\D*\d{3}\D*\d{3}\D*\d{4}\D*.+', flags=flags),
        'TABLE HEADER1': re.compile(r'^\([0-9A-Z]+\)$'),
        'TABLE HEADER2': re.compile(r'\D+.*in\s+millions\D*.*\D+$', flags=flags),
        'TABLE HEADER3': re.compile(r'\D+.*in\s+thousands\D*.*\D+$', flags=flags),
        'TABLE HEADER4': re.compile(r'\D+.*unaudited.*\D+$', flags=flags),
        'ABOUT HEADER': re.compile(r'^ABOUT .+', flags=flags),
        'NOTES': re.compile(r'^note\w{,2}\D*.+', flags=flags),
        'EXHIBIT': re.compile(r'.*Exhibit\s*\W*99.*', flags=flags),
        'FOOTNOTES': re.compile(r'^\s*\(\d+\)\s*.+', flags=flags),
        'TRADEMARKS1': re.compile('.+registered trademark.+', flags=flags),
        'TRADEMARKS2': re.compile('.+trademark.+of.+', flags=flags),
        'PAGE NUMBERS1': re.compile('^\s*\D*\d+\D*\s*$', flags=flags),
        'PAGE NUMBERS2': re.compile('\s*page\s*\d+\s*\w*\d*.+', flags=flags),
        'RELEASE NOTE1': re.compile('^this.+press.+release.+', flags=flags),
        'RELEASE NOTE2': re.compile('.+securities.+act.+of.+', flags=flags),
        'COPYRIGHTS': re.compile('.+copyright.+all rights reserved.', flags=flags),
    }
    regex_sub = lambda pattern : len(re.sub(pattern, '', text))
    return all(map(regex_sub, PATTERNS.values()))

def clean_text(er, trim_limit=0.05):
    """Cleans up the text.
    
    Params:
    -------
    
    er: string
        Text to trim
        
    trim_limit: float, Default: 5%
        We can't cut down text to under 5% of original.
        
    Returns:
    --------
        string, trimmed text
    """
    text = '\n\n'.join((text for text in re.split('\n\n', er) if clean_text_helper(text)))
    return text if len(text)/float(len(er)) >= trim_limit else er
    
def main():
    POOL_SIZE = 10
    files = filelistings() 
    pool = Pool(processes=POOL_SIZE)
    pool.map(parser_dir, files.values())
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
