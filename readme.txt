Proof-of-concept impelmentation of a linear model to classify fake and non-fake news articles. Splits the data into training and tests set, trains, tests on the test set, prints out classification report and most predictive features for each feature-type.  Types of features for a given training can be modified in the Pipeline.  

Usage: "python fake_news_detector.py fake_articles_file non_fake_articles_file"

Training/Testing Data:

	"fake_6.5k.txt" contains 6.5 articles (and feature data as descrbied below) scraped from websites tagged “BS” by the BS-detector Chrome extension, http://bsdetector.tech/ in October 2016.  Initially obtained from https://www.kaggle.com/mrisdal/fake-news.  Prepocessed to remove non-English articles, to remove any mentions of the source from the headline and body, etc., and sampled. 

	"real_6.5k.txt" contains a sample of 6.5 articles (and feature data as descrbied below) from The Guardian from June-October 2016, obtainved via the API (http://open-platform.theguardian.com/).  Contains only articles from the sections "US News", "World News", "Politics".  Prepocessed to remove any mentions of the source.


	Format of the data files:

		headline \t	headline POS tags \t headline syntactic parse \t article body \t article body POS tags \t LIWC categories counts

	POS tags obtained via NLTK POS tagger. Syntactic parse obtained via PyStatParse.  (See the helper script tagger_parser.py)
	LIWC (Linguistic Inquiry and Word Counts) counts obtained via LIWC2015 tool (https://liwc.wpengine.com/)
	The following LIWC categories were used (the order coresponds to the order of counts in the data files):

		'WC','Analytic','Clout','Authentic','Tone','WPS','Sixltr','Dic','function','pronoun','ppron','i','we','you','shehe','they','ipron','article','prep','auxverb','adverb','conj','negate','verb','adj','compare','interrog','number','quant','affect','posemo','negemo','anx','anger','sad','cogproc','insight','cause','discrep','tentat','certain','differ','informal','swear','netspeak','assent','nonflu','filler','AllPunc','Period','Comma','Colon','SemiC','QMark','Exclam','Dash','Quote','Apostro','Parenth','OtherP'



