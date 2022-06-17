# BalCompress
The code and dataset for "BalCompress: BalCompress: Mode Compression with Open-domain Unlabeled Data for Named Entity Recognition".

The repo is based on [Fliar](https://github.com/flairNLP/flair) framework with a lot of modifications.

### Requirements
This repository is tested on pytorch 1.10.2 with CUDA==10.2 and cuDNN==7.6. Please run

```
pip3 install -r requirements.txt
```

to install all dependencies


### Usage

- Train teacher (optional)


You can fine-tune a teacher with following command *or* use the fine-tuned model provided by [Fliar](https://github.com/flairNLP/flair) (repace teacher_path with 'flair/ner-english-large').

```
python train_teacher.py --save_path resources/taggers/teacher \
                        --root_path data/conll-2003\
                        --train train.txt \
                        --dev dev.txt \
                        --test test.txt
```

- rank and select unlabeled data (you must put the unlabeled data in the same folder as your train)

```
python select_data.py --teacher_path flair/ner-english-large \
                      --unlabeled_path data/wikitext/wiki_split_30.txt \
                      --save_path data/conll-2003/unlabeled.txt \
                      --number 40000
```

- Distill a student model
```
python train_student.py --save_path resources/taggers/student \
                        --root_path data/conll-2003 \
                        --train train.txt \
                        --dev dev.txt \
                        --test test.txt \
                        --unlabeled unlabeled.txt \
                        --teacher_path flair/ner-english-large
```

- Train on OntoNote 5.0

Download OntoNote 5.0 data from [LDC](https://catalog.ldc.upenn.edu/LDC2013T19), put it in the data folder, and run the commands above.
