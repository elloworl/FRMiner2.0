import itertools
import logging
import random
from collections import deque
from typing import Dict,Union, Any,Iterable, Iterator, List, Tuple, Deque, TypeVar,Callable
from typing import cast
from collections import defaultdict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import add_noise_to_dict_values
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
DataArray = TypeVar("DataArray", torch.Tensor, Dict[str, torch.Tensor])

def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields['epoch_num'] = MetadataField(epoch)
    return batch

def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        #print("instances",instance)
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        #print("padding",padding_lengths)
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
                #print("noise",noisy_lengths[field_name])
            padding_lengths = noisy_lengths
            #print(padding_lengths)

        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        #print(instance_with_lengths)

        instances_with_lengths.append(instance_with_lengths)
    #print("instances",instances_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    #for item in instances_with_lengths:
        #print(item[0])
    #print("sort_instances",instances_with_lengths)

    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this batch.  Each ``Instance``
        has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this batch into arrays of consistent length, or to set
        model parameters, etc.
        """
        padding_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        all_instance_lengths: List[Dict[str, Dict[str, int]]] = [instance.get_padding_lengths()
                                                                 for instance in self.instances]
        if not all_instance_lengths:
            return {**padding_lengths}
        all_field_lengths: Dict[str, List[Dict[str, int]]] = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)
        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x[padding_key] if padding_key in x else 0 for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value
        return {**padding_lengths}

def batch2tensor_dict(batch,
                      padding_lengths: Dict[str, Dict[str, int]] = None,
                       verbose: bool = False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # This complex return type is actually predefined elsewhere as a DataArray,
        # but we can't use it because mypy doesn't like it.
        """
        This method converts this ``Batch`` into a set of pytorch Tensors that can be passed
        through a model.  In order for the tensors to be valid tensors, all ``Instances`` in this
        batch need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the tensors for each field in each instance into a set of
        batched tensors for each field.

        Parameters
        ----------
        padding_lengths : ``Dict[str, Dict[str, int]]``
            If a key is present in this dictionary with a non-``None`` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").
        verbose : ``bool``, optional (default=``False``)
            Should we output logging information when we're doing this padding?  If the batch is
            large, this is nice to have, because padding a large batch could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious (and really slow).

        Returns
        -------
        tensors : ``Dict[str, DataArray]``
            A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
            This is a `batch` of instances, so, e.g., if the instances have a "question" field and
            an "answer" field, the "question" fields for all of the instances will be grouped
            together into a single tensor, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of tensors, for batched computation. Additionally,
            for complex ``Fields``, the value of the dictionary key is not necessarily a single
            tensor.  For example, with the ``TextField``, the output is a dictionary mapping
            ``TokenIndexer`` keys to tensors. The number of elements in this sub-dictionary
            therefore corresponds to the number of ``TokenIndexers`` used to index the
            ``TextField``.  Each ``Field`` class is responsible for batching its own output.
        """
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular field and padding key.  If we were, we use that
        # instead of the instance-based one.
        if verbose:
            logger.info("Padding batch of size %d to lengths %s", len(batch.instances), str(padding_lengths))
            logger.info("Getting max lengths from instances")
        instance_padding_lengths = batch.get_padding_lengths()
        #print("1",instance_padding_lengths)
        #print("2",padding_lengths)
        if verbose:
            logger.info("Instance max lengths: %s", str(instance_padding_lengths))
        lengths_to_use: Dict[str, Dict[str, int]] = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_lengths[field_name].get(padding_key) is not None:
                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]

        #print("lenths2use:",lengths_to_use)

        # Now we actually pad the instances to tensors.
        field_tensors: Dict[str, list] = defaultdict(list)
        if verbose:
            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))

        #######################开始padding
        for instance in batch.instances:
            for field, tensors in as_tensor_dict(instance,lengths_to_use).items():
                field_tensors[field].append(tensors)

        # Finally, we combine the tensors that we got for each instance into one big tensor (or set
        # of tensors) per field.  The `Field` classes themselves have the logic for batching the
        # tensors together, so we grab a dictionary of field_name -> field class from the first
        # instance in the batch.
        field_classes = batch.instances[0].fields
        final_fields = {}
        for field_name, field_tensor_list in field_tensors.items():
            final_fields[field_name] = field_classes[field_name].batch_tensors(field_tensor_list)
        return final_fields

def as_tensor_dict(instance,padding_lengths: Dict[str, Dict[str, int]] = None) -> Dict[str, DataArray]:
    """
    Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
    keyed by field name, then by padding key, the same as the return value in
    :func:`get_padding_lengths`), returning a list of torch tensors for each field.

    If ``padding_lengths`` is omitted, we will call ``self.get_padding_lengths()`` to get the
    sizes of the tensors to create.
    """
    '''
        set the padding/truncating length
        num_fields: the number of sentences in a dialog
        tokens_length/num_tokens: the number of words in each sentence
    '''
    custom_padding_lengths = {'num_fields': 20,'tokens_length':50,'num_tokens':50}


    padding_lengths = padding_lengths or instance.get_padding_lengths()
    tensors = {}

    for field_name, field in instance.fields.items():
        #print(field_name,field)
        if field_name == 'dialog1' or field_name == 'dialog2':
            tensors[field_name] = listField_as_tensor(field,custom_padding_lengths,padding_lengths[field_name])
        else:
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
    #assert 1==2
    return tensors

def listField_as_tensor(fields, custom_padding_lengths,
                        padding_lengths: Dict[str, int]) -> DataArray:

    sen_num = min(custom_padding_lengths['num_fields'],padding_lengths['num_fields'])
    padded_field_list = pad_sequence_to_length(fields.field_list,
                                  sen_num,
                                  fields.field_list[0].empty_field)

    # Here we're removing the scoping on the padding length keys that we added in
    # `get_padding_lengths`; see the note there for more detail.
    child_padding_lengths = {key.replace('list_', '', 1): value
                             for key, value in padding_lengths.items()
                             if key.startswith('list_')}
    for key,value in child_padding_lengths.items():
        child_padding_lengths[key] = min(value,custom_padding_lengths[key])
    #print(sen_num,child_padding_lengths)
    padded_fields = [field.as_tensor(child_padding_lengths)
                     for field in padded_field_list]
    return fields.field_list[0].batch_tensors(padded_fields)

def pad_sequence_to_length(sequence: List,
                           desired_length: int,
                           default_value: Callable[[], Any] = lambda: 0,
                           padding_on_right: bool = True) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.

    padded_sequence = sequence[:desired_length]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence


@DataIterator.register("custom_iterator")
class BucketIterator(DataIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.

        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first

    @overrides
    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.

        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        """
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            if self._cache_instances and key in self._cache:
                # Serve the results from the cache.
                tensor_dicts = self._cache[key]

                if shuffle:
                    # TODO(brendanr): How can we handle this shuffle in a way
                    # that respects multi-GPU friendly grouping?
                    random.shuffle(tensor_dicts)
                for tensor_dict in tensor_dicts:
                    if self._track_epoch:
                        # The tensor_dict already has an "epoch_num" tensor,
                        # so just fill it with the right value.
                        epoch_tensor: torch.Tensor = tensor_dict['epoch_num']
                        epoch_tensor.fill_(epoch)
                    yield tensor_dict
            else:
                batches = self._create_batches(instances, shuffle)

                # Should we add the instances to the cache this epoch?
                add_to_cache = self._cache_instances and key not in self._cache

                for batch in batches:
                    if self._track_epoch:
                        add_epoch_number(batch, epoch)

                    if self.vocab is not None:
                        batch.index_instances(self.vocab)

                    padding_lengths = batch.get_padding_lengths()

                    logger.debug("Batch padding lengths: %s", str(padding_lengths))
                    logger.debug("Batch size: %d", len(batch.instances))


                    tensor_dict = batch2tensor_dict(batch,padding_lengths)

                    if add_to_cache:
                        self._cache[key].append(tensor_dict)


                    yield tensor_dict

            # Increment epoch tracker
            self._epochs[key] = epoch + 1


    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)



            yield from batches
