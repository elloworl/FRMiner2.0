
import json
import random
import re
from collections import defaultdict
from itertools import permutations
from typing import Dict, List
import logging
import string
from allennlp.data import Field
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class Comment:
    user = None
    user_id = None
    user_type = None
    user_site_admin = None
    body = None


class Issue:
    number = None
    label = None
    owner = None
    title = None
    body = None
    comments = []


@DatasetReader.register("issue_reader_siamese")
class IssueReaderSiamese(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._class_cnt = defaultdict(int)

    def read_dataset(self, file_path):
        features = []
        others = []

        with open(cached_path(file_path), "r", encoding= 'unicode_escape') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip()
                if not line or len(line) == 0:
                    continue
                line = eval(line)
                if len(line[1]) != 0:
                  if line[-1] == 'feature':
                      features.append(line)
                  else:
                      others.append(line)
        return features, others

    @overrides
    def _read(self, file_path):
        text2instance = []
        is_evaluate = False
        iter_num = 0
        sample_features, sample_others = self.read_dataset(file_path)
        sample_data = sample_features + sample_others
        same_num = 0
        diff_num = 0
        print(len(sample_features), len(sample_others))

        if "train" in file_path:
            logger.info("Begin training-------")
            feature_num = len(sample_features)
            random.shuffle(sample_features)
            for i in range(feature_num - 1):
                for j in range(i + 1, feature_num):
                    sample = sample_features[i]
                    positive = sample_features[j]
                    text2instance.append((positive, sample))
                    same_num += 1
            print(text2instance.__len__())
            other_num = len(sample_others)
            random.shuffle(sample_others)
            for i in range(other_num - 1):
                for j in range(i + 1, other_num):
                    sample = sample_others[i]
                    negative = sample_others[j]
                    text2instance.append((negative, sample))
                    same_num += 1
            print(text2instance.__len__())
            for i in range(feature_num):
                for negative in sample_others:
                  sample = sample_features[i]
                  if i <= feature_num/2:
                    text2instance.append((negative, sample))
                  else:
                    text2instance.append((sample, negative))
                  diff_num += 1
            print(text2instance.__len__())


        elif "test" in file_path:
            logger.info("Begin evaluating-------")
            train_features, train_others = self.read_dataset(re.sub("test", "train", file_path))
            is_evaluate = True
            iter_num = 11

            for sample in sample_data:
                for i in range(iter_num):
                    positive = random.choice(train_features)
                    negative = random.choice(train_others)
                    text2instance.append((positive, sample))
                    text2instance.append((negative, sample))
                    same_num += 1
                    diff_num += 1
                rand = random.choice(train_features+train_others)
                text2instance.append((rand,sample))
            print(text2instance.__len__())

        logger.info(f"Dataset Count: {len(text2instance)}")

        logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

        # 打乱
        random.shuffle(text2instance)
        for inputs in text2instance:
            yield self.text_to_instance(inputs, is_evaluate=is_evaluate, iter_num=iter_num)


    @overrides
    def text_to_instance(self, p, is_gold=False, is_evaluate=False, iter_num=1) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        ins1, ins2 = p
        dialog = ListField([TextField([word for word in self._tokenizer.tokenize(line[1])],
                                      self._token_indexers)
                            for line in ins1[1]])
        fields['dialog1'] = dialog

        dialog = ListField([TextField([word for word in self._tokenizer.tokenize(line[1])],
                                      self._token_indexers)
                            for line in ins2[1]])
        fields['dialog2'] = dialog

        if ins1[-1] is not None and ins2[-1] is not None:
            if ins1[-1] == ins2[-1]:
                fields['label'] = LabelField("same")
            else:
                fields['label'] = LabelField("diff")
            fields['label_tags'] = LabelField("@".join([ins1[-1], ins2[-1]]), label_namespace="label_tags")
        fields['metadata'] = MetadataField({"is_gold": is_gold, "pair_instance": p})
        fields['meta_eval'] = MetadataField({"is_evaluate": is_evaluate, "dialog2_id": ins2[0], "num": iter_num})

        return Instance(fields)