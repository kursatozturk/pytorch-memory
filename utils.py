import numpy as np
from random import randint, choice
import itertools
from typing import Tuple, Iterable, List, Callable, Dict, Generator, Union
import operator
import functools
import math
from torch import nn
import torch
import time
import json
from scipy.io import wavfile


Sample = Union[np.ndarray, torch.Tensor]
MemorySamples = Iterable[Sample]
Batch = Iterable[Sample]
Epoch = Iterable[Tuple[Batch, Batch]]


class NotEnoughGPURam(Exception):
    pass

def i_arr_mul(array: Iterable) -> int:
    return functools.reduce(operator.mul, array, 1)

def KiB(coef: float = 1) -> int:
    return math.floor(coef * 1024)

def MiB(coef: float = 1) -> int:
    return math.floor(1024 * KiB(coef))

def GiB(coef: float = 1) -> int:
    return math.floor(1024 * MiB(coef))

def round_to(number: int, modulo: int) -> int:
    return (number // modulo) * modulo

def generator_with_stop_iteration(gen: Generator) -> Generator:
    try:
        return itertools.chain([next(gen)], gen)
    except StopIteration:
        raise StopIteration
    

def get_variable_count(model_class: type, input_shape: Tuple) -> int:
    """
        calculates the input & output variable count. Does not include the layer parameters.
        @param model_class must be class inherited from torch.nn.Module.
    """
    network = model_class()
    variable_count = {
            'count': 0
            }

    def inspect(_cls_instance, _inp, _out):
        """
            Violates functional paradigm,
            but easier to inspect complex networks.
        """
        output_variable_count = i_arr_mul(_out.data.size())
        grad_variable_count = 0
        if _out.requires_grad:
            grad_variable_count = output_variable_count
        variable_count['count'] += grad_variable_count + output_variable_count

    for layer in network.modules():
        layer.register_forward_hook(inspect)
    fake_input = torch.zeros(input_shape)
    out = network.forward(fake_input)
    del network
    del fake_input
    del out
    return variable_count['count']

def get_parameter_count(model_class: type):
    """
        Calculates the parameters used in the layers of network.
        Does not include input&output variables.

        @param model_class must be class inherited from torch.nn.Module.
    """
    network = model_class()
    modules = network.modules()
    next(modules) # discard model itself

    def parameter_count_of_layer(layer: torch.nn.Module) -> int:
        return sum(i_arr_mul(param.shape) for param in layer.parameters())
    
    total_parameter_count = sum(map(parameter_count_of_layer, modules))
    del network
    return total_parameter_count

def summary_of_network(model_class: type, input_shape: Tuple) -> List[Dict]:
    """
        Returns the summary of the model as list for each dict is a summary of the corresponding layer
        @param model_class must be class inherited from torch.nn.Module.
    """
    network_summary = []
    network = model_class()
    
    def inspect(_cls_instance, _inp, _out):
        """
            Violates functional paradigm,
            but easier to inspect complex networks.
        """
        summary = {
                    'input_shape': tuple(_inp[0].data.size()),
                    'output_shape': tuple(_out.data.size()),
                    'parameter_count': sum(map(i_arr_mul, (x.shape for x in _cls_instance.parameters()))),
                    'layer_type_name': _cls_instance.__class__.__name__
                }
        network_summary.append(summary)

    for layer in network.modules():
        layer.register_forward_hook(inspect)
    fake_input = torch.zeros(input_shape)
    out = network.forward(fake_input)
    del network
    del fake_input
    del out
    return network_summary
 

class Config:
    def __init__(self, 
            cpu_memory_limit: int, # in bytes
            gpu_memory_limit: int, # in bytes
            input_shape: Tuple,
            output_shape: Tuple,
            model_class: type,
            total_sample_count: int,
            input_dtype: type = np.float64,
            output_dtype: type = np.float64,
            one_read_count: int = 1,
            preferred_batch_size: int = None
            ):
        """
            @param cpu_memory_limit: Limit of the memory that a  sample batch is allowed to use in CPU RAM.
            @param gpu_memory_limit: Limit of the memory that a input&output batch is allowed to use in GPU RAM.
            @param input_shape: Input shape of a sample. 
            @param output_shape: Output_shape of a sample.
            @param model_class: Class of the network to be tarined.
            @param total_sample_count: total sample count in the memory.
            @param input_dtype: data type of the input samples.
            @param output_dtype: data type of the output samples.
            @param one_read_count: Count of sample generated by read_input.
            @param preferred_batch_size: if it is applicable, preferred_batch_size - preferred_batch_size % one_read_count will be used as batch size
        """
        self.cpu_memory_limit = cpu_memory_limit # bytes.
        self.gpu_memory_limit = gpu_memory_limit # bytes.
        self.total_sample_count = total_sample_count
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.one_read_count = one_read_count
        try:
            self.input_item_size = self.input_dtype().itemsize
            self.output_item_size = self.output_dtype().itemsize
        except Exception as e:
            print(f'INVALID DTYPE GIVEN!. {e!r}')
            print('elem size is set to default (4 bytes)')
            self.input_item_size = 4
            self.output_item_size = 4
        self.parameter_count = get_parameter_count(model_class=model_class)
        self.total_intermediate_variable_count = get_variable_count(model_class=model_class, input_shape=input_shape)
        self.preferred_batch_size = preferred_batch_size or (self.gpu_memory_limit - self.model_memory_usage) // self.single_sample_memory_usage
        if self.model_memory_usage >= self.gpu_memory_limit:
            raise NotEnoughGPURam('Parameters exhausts GPU Ram. Please either increase GPU limit, or shrink network.')
        if self.batch_size <= 0:
            raise NotEnoughGPURam('Not Enough Space in GPU for a single batch!')

    @property
    def input_size(self):
        return self.input_item_size * i_arr_mul(self.input_shape)

    @property
    def output_size(self):
        return self.output_item_size * i_arr_mul(self.output_shape)

    @property
    def sample_size(self):
        return self.input_size + self.output_size

    @property
    def single_sample_memory_usage(self):
        return self.sample_size + self.total_intermediate_variable_count * self.input_item_size
    
    @property
    def model_memory_usage(self):
        return self.parameter_count * self.input_item_size

    @property
    def batch_size(self):
        return round_to(
                max(
                    min(
                        (self.gpu_memory_limit - self.model_memory_usage) // self.single_sample_memory_usage,
                        self.preferred_batch_size,
                        self._allowed_sample_count * self.one_read_count
                        ),
                    self.one_read_count
                    ),
                self.one_read_count
                )

    @property
    def sample_count_in_single_batch(self):
        return self.batch_size // self.one_read_count

    @property
    def sample_batch_memory_footprint(self):
        return self.batch_size * self.sample_size

    @property
    def _allowed_sample_count(self):
        return min(self.total_sample_count, self.cpu_memory_limit // (self.sample_size * self.one_read_count))

    @property
    def allowed_sample_count(self):
        return round_to(self._allowed_sample_count, self.sample_count_in_single_batch) 

    @property
    def batch_count(self):
        return self.total_sample_count // self.sample_count_in_single_batch

    @property
    def total_read_count(self):
        return self.total_sample_count // self.allowed_sample_count + (1 if self.total_sample_count % self.allowed_sample_count else 0)

    @property
    def batch_count_in_memory(self):
        return self.allowed_sample_count // self.sample_count_in_single_batch
 
class Feeder:
    """
        This is the sample file feeder for the deep learning model.
        
    """
    path = '/home/kursat/Desktop/Projects/Menrva/Dataset/nsynth-valid.jsonwav/nsynth-valid'

    def __init__(
            self,
            configs: Config, 
            input_list: List[str], 
            output_list: List[str],
            read_input: Callable[[str], Iterable[np.ndarray]],
            get_output: Callable[[str], Iterable[np.ndarray]],
            apply_on_input_batch: Callable[[Iterable[np.ndarray]], np.ndarray] = None,
            apply_on_output_batch: Callable[[Iterable[np.ndarray]], np.ndarray] = None,
            ) -> Iterable[np.ndarray]:
        """
            Reads files from hard drive and returns,
            by not violating the memory restrictions
            batch by batch.
            i.e. if memory limit is 1gb, 
            calculates memory_footprint = 1gb - sample_batch_memory_footprint(batch_size)
            Returns a generator that reads 'memory_footprint bytes' of music in each pass.
            @param config: Config instance for resource limitations and requirements,
            @param input_list: List of input paths that to be read from disk,
            @param output_list: List of output paths that to be read from disk,
            @param read_input: A Callable taking one parameter of input path. Should return an iterator containin np.ndarray items,
            @param get_output: A Callable taking one parameter of output path. Should return an iterator containin np.ndarray items,
            @param apply_on_input_batch: A Callable taking one parameter of input batch. Should return a np.ndarray,
            @param apply_on_output_batch: A Callable taking one parameter of output batch. Should return a np.ndarray,
        """
        if apply_on_input_batch is None:
            apply_on_input_batch = lambda x: np.array(list(x), dtype=configs.input_dtype)
        if apply_on_output_batch is None:
            apply_on_output_batch = lambda x: np.array(list(x), dtype=configs.output_dtype)
        self.apply_on_input_batch = apply_on_input_batch
        self.apply_on_output_batch = apply_on_output_batch
        self.read_input = read_input
        self.get_output = get_output
        self.input_list = input_list
        self.output_list = output_list
        self.configs = configs

    def _get_samples_one_pass(self) -> Iterable[np.ndarray]:
        step = self.configs.allowed_sample_count
        for i in range(self.configs.total_read_count):
            samples_on_memory = (
                    inp for inp_path in self.input_list[
                        i * step: (i + 1) * step
                        ]
                    for inp in self.read_input(inp_path)
                    )
            labels_on_memory = (
                    out for out_path in self.output_list[
                        i * step: (i + 1) * step
                        ]
                    for out in self.get_output(out_path)
                    )
            for batch in range(self.configs.batch_count_in_memory):
                yield self.apply_on_input_batch(
                        (sample for sample in itertools.islice(samples_on_memory, self.configs.batch_size))
                        ), self.apply_on_output_batch(
                        (label for label in itertools.islice(labels_on_memory, self.configs.batch_size))
                        )

    def generate_epochs(self, epoch_count:int = 1) -> Iterable[Iterable[np.ndarray]]:
        samples_generator = self._get_samples_one_pass()
        if epoch_count == 1:
            yield samples_generator
        else:
            yield from itertools.tee(samples_generator, epoch_count)

def generate_epoch(
        configs: Config, 
        input_list: List[str], 
        output_list: List[str],
        read_input: Callable[[str], Iterable[np.ndarray]],
        get_output: Callable[[str], Iterable[np.ndarray]],
        apply_on_input_batch: Callable[[Iterable[np.ndarray]], np.ndarray] = None,
        apply_on_output_batch: Callable[[Iterable[np.ndarray]], np.ndarray] = None,
        ) -> Epoch:
        """
            It reads samples batch by batch and generates a generator that 
            returns a batch of sample that to be used as an epoch in training 
            process.
            calculates memory_footprint = 1gb - sample_batch_memory_footprint(batch_size)
            Returns a generator that reads 'memory_footprint bytes' of music in each pass.
            @param config: Config instance for resource limitations and requirements,
            @param input_list: List of input paths that to be read from disk,
            @param output_list: List of output paths that to be read from disk,
            @param read_input: A Callable taking one parameter of input path. Should return an iterator containin np.ndarray items,
            @param get_output: A Callable taking one parameter of output path. Should return an iterator containin np.ndarray items,
            @param apply_on_input_batch: A Callable taking one parameter of input batch. Should return a np.ndarray,
            @param apply_on_output_batch: A Callable taking one parameter of output batch. Should return a np.ndarray,
        """
        if apply_on_input_batch is None:
            apply_on_input_batch = lambda x: np.array(list(x), dtype=configs.input_dtype)
        if apply_on_output_batch is None:
            apply_on_output_batch = lambda x: np.array(list(x), dtype=configs.output_dtype)
        samples = map(read_input, input_list)
        labels = map(get_output, output_list)
        

        def load_to_memory(_samples: Iterable) -> MemorySamples:
            loaded_samples = itertools.islice(_samples, configs.allowed_sample_count)
            try:
                yield generator_with_stop_iteration(loaded_samples)
                yield from load_to_memory(_samples)
            except StopIteration:
                pass
        
        in_memory_samples = load_to_memory(samples)
        in_memory_labels = load_to_memory(labels)
        
        def read_batch(_samples: List[Sample]) -> Batch:
            batch_samples = itertools.islice(_samples, configs.sample_count_in_single_batch)
            batch = (b for sample in batch_samples for b in sample)
            try:
                yield generator_with_stop_iteration(batch)
                yield from read_batch(_samples)
            except StopIteration:
                pass
        
        batched_samples = map(read_batch, in_memory_samples)
        batched_labels = map(read_batch, in_memory_labels)
        
        def unfold_batch(_samples: Iterable[Generator[Batch, None, None]]) -> Iterable[Batch]:
            yield from (batch for batches in _samples for batch in batches)
        
        yield from zip(
                map(apply_on_input_batch, unfold_batch(batched_samples)),
                map(apply_on_output_batch, unfold_batch(batched_labels))
                )

def multiply_epochs(epoch_generator: Generator[Sample, None, None], epoch_count: int = 2) -> Generator[Generator[Sample, None, None], None, None]:
    yield from itertools.tee(epoch_generator, epoch_count)

