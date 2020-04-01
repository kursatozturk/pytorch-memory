## Pytorch Utilities

Working with big amount of data, or models having millions of parameters is a great challenge if you have limited resources.

I tried to solve this memory issues. 


### Config
This is the configuration class for the feeder. 
It calculates: 
* How many sample can be hold in the memory,
* Maximum batch size that can be used to train/test the network.

It is required for the Feeder class.

#### Usage: 
```
config = Config(
	cpu_memory_limit=GiB(2),
	gpu_memory_limit=GiB(1.5),
	total_sample_count=2048,
	input_shape=(3, 1024, 1024),
	output_shape=(256,)
	model_class=model_class,
	input_dtype=np.float32,
	output_dtype=np.float32,
	one_read_count=2,
	preferred_batch_size=16
)
```

most of the parameters are self-explanatary.
one_read_count is a bit tricky here. In Feeder class and the generate_epoch function, inputs are read by a registered hook, input_read. Sometimes, training can require to use a single sample multiple times with small variations like adding noise or shifting etc. Here you can read the sample, add some variance to it and yield both of them. Config need to know exact count to be able to calculate correct estimations.
preferred_batch_size is also self-explanatary however preferred means that it is not a must. If memory is not available or if it is not a multiple of one_read_count, it will be recalculated.

## Feeder

Feeder class is responsible to implement the generators by looking the config. It does not read any data, it does not alter any data, it does not reshape any data. Rather all related functions are given as parameter to Feeder. Feeder just uses Config's estimations to create data feeder generators.

### Usage
```
feeder = Feeder(
	configs=config,
	input_list, # (input_identities). note that not the actual input samples, identities that read_input function will be able to read the sample by using that identity.
	output_list, # same as input list except for output.
	read_input, # callable object takes the identity [str] from input_list as input and returns a generator object of numpy.ndarray/torch.Tensor objects.
	get_output, # same as read_input except for output.
	apply_on_input_batch, # callable object takes iterable of numpy.ndarray/torch.Tensor and returns numpy.ndarray/torch.Tensor. It is useful for cases such as to move batch from cpu to gpu, reshaping etc.
	apply_on_output_batch, # same as apply_on_input_batch except for output.
	)


epochs = feeder.generate_epochs(epoch_count=100)
for epoch in epochs:
	for x_batch, y_batch in epoch:
		y_pred = model.forward(x_batch)
		calculate_loss(y_batch, y_pred)
		...
```

## generate_epoch

Does the exact same thing with Feeder with little variations, such as it reads all of the data unlike Feeder, which reads only multiple of batch size, i.e if there are remainings samples less than batch size, Feeder discards them.

It is a * functional approach * for the problem. I evaluated both, and saw that Feeder is slightly faster than generate_epochs and use slightly less memory. However generate_epoch is easy to read and debug. I will continue to improve both methods.
### Usage
```
epoch_generator = generate_epoch(
		configs=config,
		input_list, # (input_identities). note that not the actual input samples, identities that read_input function will be able to read the sample by using that identity.
		output_list, # same as input list except for output.
		read_input, # callable object takes the identity [str] from input_list as input and returns a generator object of numpy.ndarray/torch.Tensor objects.
		get_output, # same as read_input except for output.
		apply_on_input_batch, # callable object takes iterable of numpy.ndarray/torch.Tensor and returns numpy.ndarray/torch.Tensor. It is useful for cases such as to move batch from cpu to gpu, reshaping etc.
		apply_on_output_batch, # same as apply_on_input_batch except for output.
		)
epochs = multiply_epochs(epoch_generator=epoch_generator, epoch_count=100)
for epoch in epochs:
	for x_batch, y_batch in epoch:
		y_pred = model.forward(x_batch)
		calculate_loss(y_batch, y_pred)
		...

```


It basically utilizes the generators. This way, it creates pseudo-threaded implementation and decreases memory usage dramatically since all the operations are handled on the go. With the help of caching, it may even makes use of faster reading after first epoch finished. However it highly depends on data size and the os/build. 

Note: After diving into memory usage further I discovered that script uses very high amount of memory, way higher than given limit. It is due to the python's memory management system however I cannot decrease it further with current implementatios. Still limits makes sense, so use it with caution that it will use a bit higher than given.
