from flair.data import Corpus
from flair.datasets import CONLL_03, ColumnCorpus, ColumnDataset
from flair.embeddings import WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import argparse
import torch
import flair
import os
from copy import deepcopy
import pdb
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument('--save_path', type=str, required=True, help="model save path")
parser.add_argument('--root_path', type=str, required=True, help="Root path to data")
parser.add_argument('--train', type=str, required=True, help="Training corpus path.")
parser.add_argument('--dev', type=str, required=True, help="Validation corpus path.")
parser.add_argument('--test', type=str, required=True, help="Testing corpus path.")
parser.add_argument('--unlabeled', type=str, required=True, help='unlabeled path')
parser.add_argument('--teacher_path', type=str, required=True, help='teacher path')


parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size.")
parser.add_argument("--epoch", type=int, default=200, help="Total epoch.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.3, help="learning rate")


args = parser.parse_args()

unlabeled_file_name = args.unlabeled.split('/')[-1]


columns = {0: 'text', 1: 'ner'}


corpus = ColumnCorpus(
    args.root_path, columns,
    train_file=f'{unlabeled_file_name}',
    dev_file=f'{args.dev}',
    test_file=f'{args.test}'
)


aug_dataset = ColumnDataset(os.path.join(args.root_path, args.train), columns)
print(len(aug_dataset))

label_type = 'ner'

tagger = SequenceTagger.load(args.teacher_path)
tagger.eval()
with torch.no_grad():
    for sentence in tqdm(aug_dataset):
        tagger.predict(sentence)
        tagger.assign_softtarget(sentence)

    for sentence in tqdm(corpus.train):
        tagger.predict(sentence)
        tagger.assign_softtarget(sentence)

label_dict = deepcopy(tagger.tag_dictionary)

embedding_types = [
    WordEmbeddings('en'),
    CharacterEmbeddings(char_embedding_dim=25, hidden_size_char=25)
]

embeddings = StackedEmbeddings(embeddings=embedding_types)


tagger = SequenceTagger(hidden_size=args.hidden_size,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)

# trainer.distill_balanced_batch(
#     # 'test',
#     learning_rate=0.3,
#     mini_batch_size=32,
#     monitor_test=True,
#     max_epochs=200,
#     use_final_model_for_eval=False,
#     embeddings_storage_mode='gpu',
#     aug_dataset=aug_dataset,
#     two_loss=False
# )

trainer.train_balanced_batch(
    args.save_path,
    learning_rate=args.lr,
    mini_batch_size=args.batch_size,
    monitor_test=True,
    max_epochs=args.epoch,
    use_final_model_for_eval=False,
    embeddings_storage_mode='gpu',
    aug_dataset=aug_dataset,
)