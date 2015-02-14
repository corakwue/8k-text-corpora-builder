## 8-K text corpra builder

For use in conjuction with [sec-edgar](https://github.com/corakwue/sec-edgar) to build categorized text corpora 
especially for earnings releases from Form 8-Ks downloaded from SEC Edgar Database.

### Sentiment Categorization

Sentiment Categorization is based on market return N-days (`TIMESPAN`) after filing using historical close price from [Xantos](corakwue.github.io/xantos) platform.

`LIMIT` parameter is used to decide sentiment based market return over `TIMESPAN` interval as `neg`, `pos`, or `neutral` (depending on desired number of class).

`adjust_delay` is used to adjust `TIMESPAN` interval and account for fact that some filings can be delayed, whose default value is 4. See [here](See http://www.sec.gov/answers/form8k.htm) for more information.


Two classes of categorization are supported:

* **2-class:** `neg` or `pos`
* **3-class:** `neg`, `pos`, or `neutral` where `neutral` indicates lack of sentiment.

## Dependencies:
* Python (2.7 or 3.3)
* Logbook
* Pandas 
* bs4 (BeautifulSoup)
* NLTK
* [Xantos](corakwue.github.io/xantos) --optional: can use Pandas instead

## Issues:
This is a hack script for personal use and not up to par with prod.
If you have any issues, feel free to contact me @ chukwuchebem.orakwue@gmail.com

For sentiment analysis, I'd warn you'd need a good:

* Sentence tokenizer
* Feature selection algorithm to train your classifier (ha!) 

I found NLTK default sentence tokenizer inadequate due to crudeness of raw text in corpus.
