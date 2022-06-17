from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import argparse


parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument('--save_path', type=str, required=True, help="model save path")
parser.add_argument('--root_path', type=str, required=True, help="Root path to data")
parser.add_argument('--train', type=str, required=True, help="Training corpus path.")
parser.add_argument('--dev', type=str, required=True, help="Validation corpus path.")
parser.add_argument('--test', type=str, required=True, help="Testing corpus path.")

args = parser.parse_args()

corpus = ColumnCorpus(
    f'{args.root_path}', 
    {0: 'text', 1: 'ner'},
    train_file=f'{args.train}',
    dev_file=f'{args.dev}',
    test_file=f'{args.test}'
)

label_type = 'ner'

label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type='ner',
    use_crf=True,
    use_rnn=False,
    reproject_embeddings=False,
)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune(
    f'{args.save_path}',
    learning_rate=5.0e-6,
    mini_batch_size=4,
    mini_batch_chunk_size=1,
)