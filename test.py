from utils import Config, Feeder, GiB, MiB, KiB, generate_epoch, multiply_epochs, summary_of_network
from scipy.io import wavfile
import json
import numpy as np
import time
import itertools
from models import MusicalSoundSegmentation
from typing import Generator

def test():
    main_path = '/home/kursat/Desktop/Projects/Menrva/49x-menrva/nsynth-test'
    file_name = f'{main_path}/examples.json'
    with open(file_name, 'r') as f:
        raw_labels = json.load(f)

    def is_clean(key):
        label = raw_labels[key]
        return not (label['qualities'][3] or \
        label['qualities'][5] or label['qualities'][9] or \
        label['instrument_family'] == 10)
    
    labels = list(filter(is_clean, raw_labels))
    
    def read_input(path):
        _, sample = wavfile.read(f'{main_path}/audio/{path}.wav')
        yield from np.split(sample, 200, axis=0)

    def get_output(path):
        label = raw_labels.get(path)
        frequencies = np.zeros((127,))
        midi_number = int(label.get('pitch'))
        freq = 2 ** (midi_number / 12)
        frequencies[:] = freq
        yield from (frequencies for _ in range(4))

    def identity(x): return list(x)

    def apply_on_batch(batch):
        try:
            l = list(batch)
            y = np.array(l, dtype=np.float32)
            return y
        except TypeError as e:
            print(e)
    
    def evaluate_epoch_time(epoch_generator: Generator) -> float:
        t = time.time()
        sample_count = 0
        epoch_count = 0
        batch_size = 0
        for epoch in epoch_generator:
            epoch_count += 1
            batch_count = 0
            for X, _ in epoch:
                batch_count += 1
                sample_count += X.shape[0]
        return time.time() - t, sample_count, batch_count, epoch_count

    config = Config(
            cpu_memory_limit=MiB(64),
            gpu_memory_limit=GiB(4),
            total_sample_count=len(labels),
            input_shape=(1, 1, 320),
            output_shape=(1, 127),
            model_class=MusicalSoundSegmentation,
            input_dtype=np.float32,
            output_dtype=np.float32,
            one_read_count=200,
            preferred_batch_size=200
            )
    feeder = Feeder(
            configs=config,
            input_list=labels,
            output_list=labels,
            read_input=read_input,
            get_output=get_output,
            apply_on_input_batch=apply_on_batch,
            apply_on_output_batch=apply_on_batch
            )
    epochs = feeder.generate_epochs(epoch_count=1000)
    cls_t, cls_sample_count, cls_batch_count, cls_epoch_count = evaluate_epoch_time(epochs) 
    epoch_generator = generate_epoch(
            configs=config,
            input_list=labels,
            output_list=labels,
            read_input=read_input,
            get_output=get_output,
            )
    epochs = multiply_epochs(epoch_generator=epoch_generator, epoch_count=1000)
    func_t, func_sample_count, func_batch_count, func_epoch_count = evaluate_epoch_time(epochs)
    print('tests done for the following network')
    model_summary = summary_of_network(MusicalSoundSegmentation, config.input_shape)
    for layer in model_summary:
        print(f'{str(layer.get("layer_type_name"))} {layer.get("input_shape")}, {layer.get("output_shape")}, {layer.get("parameter_count")}')
    print(f'Functional implementation, read {func_sample_count} sample in {func_epoch_count} epochs in {func_batch_count} batches in {func_t:.5f} seconds')
    print(f'Object Oriented Based implementation, read {cls_sample_count} sample in {cls_epoch_count} epochs in {cls_batch_count} batches in {cls_t:.5f} seconds')

if __name__ == '__main__':
    test()

