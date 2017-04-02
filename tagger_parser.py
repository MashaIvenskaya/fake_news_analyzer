import nltk
from stat_parser import Parser
import sys  
"""Helper script to add POS tags (headline and body) and syntactic parse (headline only) to a file of training data"""
"""First argument is the input file, second argument is the output file"""

reload(sys)  
sys.setdefaultencoding('utf8')

parse_dict = {}
parser = Parser()

input_file = sys.argv[1]
file_with_feats = sys.argv[2]

with open(input_file, 'rU') as f:
	with open(file_with_feats, 'w') as w:
		i = 1
		for line in f:
			print(i)
			line = line.strip()
			art_id = str(i)
			try:
				art_title, art_text, source, source_type = line.split('\t')
				try:
					title_tokens = nltk.word_tokenize(art_title)
					title_pos_tags = [x[1] for x in nltk.pos_tag(title_tokens)]
				except:
					title_pos_tags = ['n/a']
				try:
					text_tokens = nltk.word_tokenize(art_text)
					text_pos_tags = [x[1] for x in nltk.pos_tag(text_tokens)]
				except:
					text_pos_tags = ['n/a']
				try:
					title_parse = parser.parse(art_title)
					splitted = str(title_parse).split()
					title_flat_tree = ' '.join(splitted)
				except:
					title_flat_tree = 'n/a'
				output = [art_id, art_title, ' '.join(title_pos_tags), title_flat_tree, art_text, ' '.join(text_pos_tags), source, source_type]
				w.write('\t'.join(output))
			except:
				print('not enough values')
			i+=1
			w.write('\n')	
print(unknown)



  
