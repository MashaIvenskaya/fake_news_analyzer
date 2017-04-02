import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from operator import itemgetter
from collections import defaultdict
from nltk import Tree
import string
import re
import sys  

"""A hack to enforce utf8 encoding"""
reload(sys)  
sys.setdefaultencoding('utf8')

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class Baseline_Feats(BaseEstimator, TransformerMixin):
    """Extracts features from each document for baseline classifier"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        baseline_stats = []
        punctuations = list(string.punctuation)
        additional_punc = ['``', '--', '\'\'']
        punctuations.extend(additional_punc)
        
        for text in texts:
            stats = {}
            lengths = []
            try:
                sents = nltk.sent_tokenize(text)
                for sent in sents:
                    lengths.append(len(sent))
            except:
                pass
            #unkonezable sentences are not included in the average
            if len(lengths) == 0:
                avg_len = 0
            else:
                avg_len = float(sum(lengths))/len(lengths)
            stats['sent_length'] = avg_len

            caps_pattern = re.compile(r'[A-Z]+\s[A-Z]{2,}')
            caps_match = re.search(caps_pattern, text)
            if caps_match:
                stats['all_caps'] = 1
            else:
                stats['all_caps'] = 0

            running_punct_pattern = re.compile(r'[?!]{2,}|[.]{4,}')
            punct_match = re.search(running_punct_pattern, text)
            if punct_match:
                stats['running_punct'] = 1
            else:
                stats['running_punct'] = 0

            smiley_pattern = re.compile(r'(:\(|:\))+')
            smiley_match = re.search(smiley_pattern, text)
            if smiley_match:
                stats['emoticons'] = 1
            else:
                stats['emoticons'] = 0

            if text.endswith('?'):
                stats['question'] = 1
            else:
                stats['question']= 0

            baseline_stats.append(stats)
        return baseline_stats

class POS_Stats(BaseEstimator, TransformerMixin):
    """Extract POS features from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, pos_fields):
        pos_stats = []
        punctuations = list(string.punctuation)
        additional_punc = ['``', '--', '\'\'']
        punctuations.extend(additional_punc)
        for field in pos_fields:
            pos_tags = defaultdict(int)
            tags = field.split(' ')
            
            for tag in tags:
                if tag not in punctuations:
                    pos_tags[tag]+=1
            pos_stats.append(pos_tags)
        return pos_stats

class Parsing_Stats(BaseEstimator, TransformerMixin):
    """Extracts syntax features from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, parses):
        parsing_stats = []
        for parse in parses:
            parse_dict = defaultdict(int)
            if parse == 'n/a':
                parse_dict['n/a'] += 1
            else:
                try:
                    parse = Tree.fromstring(parse)
                    productions = parse.productions()
                    for production in productions:
                        if production.is_nonlexical():
                            parse_dict[str(production)]+=1
                except:
                    parse_dict['n/a'] += 1

            parsing_stats.append(parse_dict)
        return parsing_stats

class LIWC_Stats(BaseEstimator, TransformerMixin):
    """Extracts LIWC features from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, liwc_fields):
        
        liwc_category_names = ['WC','Analytic','Clout','Authentic','Tone','WPS','Sixltr','Dic','function','pronoun','ppron','i','we','you','shehe','they','ipron','article','prep','auxverb','adverb','conj','negate','verb','adj','compare','interrog','number','quant','affect','posemo','negemo','anx','anger','sad','cogproc','insight','cause','discrep','tentat','certain','differ','informal','swear','netspeak','assent','nonflu','filler','AllPunc','Period','Comma','Colon','SemiC','QMark','Exclam','Dash','Quote','Apostro','Parenth','OtherP']
        liwc_stats = []
        print('transforming liwc')
        for field in liwc_fields:
            liwc_dict = defaultdict(int)
            for i, val in enumerate(field):
                feat_name = liwc_category_names[i]
                value = float(val)
                liwc_dict[feat_name] = value
            liwc_stats.append(liwc_dict)
        return liwc_stats

class HeadlineBodyFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts the components of each post in the data: headline, body, POS tags, syntactic parse, LIWC category values. """
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = np.recarray(shape=(len(posts),), dtype=[('headline', object), ('headline_pos', object), ('headline_parse', object), ('body', object), ('body_pos', object), ('liwc', object)])
        for i, text in enumerate(posts): 
            fields = text.split('\t')
            headline = fields[1]
            headline_pos = fields[2]
            headline_parse = fields[3]
            body = fields[4]
            body_pos = fields[5]
            liwc = fields[7:]
            features['headline'][i] = headline
            features['body'][i] = body
            features['headline_pos'][i] = headline_pos
            features['headline_parse'][i] = headline_parse
            features['body_pos'][i] = body_pos
            features['liwc'][i] = liwc
        return features


pipeline = Pipeline([
    # Extract the subject & body
    ('HeadlineBodyFeatures', HeadlineBodyFeaturesExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #Pipeline for pulling features from articles
            ('headline_baseline', Pipeline([
                ('selector', ItemSelector(key='headline')),
                ('stats', Baseline_Feats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            ('body_baseline', Pipeline([
                ('selector', ItemSelector(key='body')),
                ('stats', Baseline_Feats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            ('subj_pos_stats', Pipeline([
                ('selector', ItemSelector(key='headline_pos')),
                ('stats', POS_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            ('body_pos_stats', Pipeline([
                ('selector', ItemSelector(key='headline_pos')),
                ('stats', POS_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

             ('pos_bigrams_title', Pipeline([
                 ('selector', ItemSelector(key='headline_pos')),
                 ('vect', CountVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b', max_df = 0.5)),
             ])),

             ('pos_bigrams_text', Pipeline([
                 ('selector', ItemSelector(key='headline_pos')),
                 ('vect', CountVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b', max_df = 0.5)),
             ])),

            ('parse_stats', Pipeline([
                ('selector', ItemSelector(key='headline_parse')),
                ('stats', Parsing_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            ('liwc_stats', Pipeline([
                ('selector', ItemSelector(key='liwc')),
                ('stats', LIWC_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
        ],
    )),

    # Use a SVC or Logistic Regression classifier on the combined features
    ('svc', LinearSVC(C=1.0)),
    #('logreg', LogisticRegression(penalty="l2", C=1.5, dual = True,  class_weight="auto")),
])

def show_most_informative_features(vect, clf, text=None, n=20):
    vectorizer = vect
    classifier = clf

    if text is not None:
        tvec = model.transform([text]).toarray()
    else:
        tvec = classifier.coef_
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])
    positive_dict = {}
    negative_dict = {}
    for (cp, fnp), (cn, fnn) in topn:
        try:
            positive_dict[cp] = fnp
            negative_dict[cn] = fnn
        except:
            pass
    print('TOP POSITIVE')
    for each in sorted(positive_dict.keys(), reverse=True):
        print(each, positive_dict[each])
    print('TOP NEGATIVE')
    for each in sorted(negative_dict.keys()):
        print(each, negative_dict[each])

def main():
    fake_articles_file= sys.argv[1]
    non_fake_articles_file = sys.argv[2]
                    
    with open(fake_articles_file, 'rU') as f:
        fake_articles = ['\t'.join(art.strip().split('\t')[:7]+art.strip().split('\t')[8:]) for art in f.readlines() if len(art.split('\t'))>5]

    with open(non_fake_articles_file, 'rU') as f:
        non_fake_articles = [art for art in f.readlines() if len(art.split('\t'))>5]

    news_data = fake_articles + non_fake_articles
    news_target = [1] * len(fake_articles) + [0] * len(non_fake_articles)
    print('splitting')
    X_train, X_test, y_train, y_test = train_test_split(news_data, news_target, test_size=0.4, random_state=0)
    print('fitting')
    pipeline.fit(X_train, y_train)
    print('predicting')
    y = pipeline.predict(X_test)
    print('generating report')
    print(classification_report(y, y_test))

    """Print out most informative features"""
    clf = pipeline.named_steps['svc']
    labels = [0,1]
    for i in range(len(pipeline.named_steps['union'].transformer_list)):
        features = pipeline.named_steps['union'].transformer_list[i]
        print(features[0] + ' top features:')
        vect = features[1].named_steps['vect']
        show_most_informative_features(vect, clf)
        print('\n')

if __name__ == "__main__":
    main()