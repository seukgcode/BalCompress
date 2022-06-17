from tqdm.notebook import tqdm
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from collections import defaultdict
import argparse
import pdb


parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument('--teacher_path', type=str, required=True, help="teacher model path")
parser.add_argument('--unlabeled_path', type=str, required=True, help="unlabeled data path")
parser.add_argument('--save_path', type=str, required=True, help="save path")
parser.add_argument('--number', type=int, default=50000, required=True, help="how many samples do you want?")

args = parser.parse_args()

tagger = SequenceTagger.load(f'{args.teacher_path}')
path = args.unlabeled_path

data = []

with open(path, 'r') as f:
    for idx, line in tqdm(enumerate(f)):
        if len(line.split(' ')) < 10: continue
        sentence = Sentence(line)
        
        tagger.predict(sentence)
        if len(sentence.get_labels('ner')) < 2:
            continue
        data.append(sentence)

        if data and len(data) % (args.number * 1.5 // 100) == 0:
            partion = len(data) / (args.number * 1.5) * 100 
            print(f'{partion}% sentences have been tagged.')

        if data and len(data) % (args.number * 1.5) == 0:
            break

ief = defaultdict(int)

for sent in data:
    for entity in sent.to_dict('ner')['entities']:
        ief[entity['labels'][0].value] += 1

ief = {k: len(data)/v for k, v in ief.items()}

for i in range(len(data)):
    sent = data[i]
    ef = 0
    for entity in sent.to_dict('ner')['entities']:
        ef += ief[entity['labels'][0].value]
    data[i] = (sent, ef / len(sent))

data.sort(key=lambda a: -a[1])

with open(args.save_path, 'w') as fw:
    for sentence in data[:args.number]:
        for i in sentence[0]:
            text, tag = i.text, i.get_tag('ner').value
            fw.write(f'{text} {tag}\n')
        fw.write('\n')