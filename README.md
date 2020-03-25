## Pytorch Memory Utility

Working with big amount of data, or models having millions of parameters is a great challenge if you have limited resource.

I tried to solve this memory issues. 

There exists three utility classes.

### InspectNetwork
calculates the estimated memory usage for the given network.


##### note: It assumes the network should use the forward method as this  
```
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        modules = self.modules()
        next(modules)
        return reduce(lambda inp, layer: layer(inp), modules, batch)
```

#### Usage:
```
inspector = InspectNetwork(network, input_shape)
parameter_count = inspector.get_parameter_count()
intermediate_variable_count = inspector.get+intermediate_variable_count()
```
Note that this only returns variable count. To find the memory usage, you should mutliply it with the element size. (i.e. 2 for half, 4 for float)

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
	network=model,
	input_dtype=np.float32,
	output_dtype=np.float32,
	one_read_count=2,
	preferred_batch_size=16
)
```

most of the parameters are self-explanatary.
one_read_count is a bit tricky here. In Feeder class, inputs are read by a registered hook, input_read. Sometimes, training can require to use a single sample multiple times with small variations like adding noise or shifting etc. Here you can read the sample, add some variance to it and yield both of them. Config need to know exact count to be able to calculate correct estimations.
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

It basically utilizes the generators. This way, it creates pseudo-threaded implementation and decreases memory usage dramatically since all the operations are handled on the go. With the help of caching, it may even makes use of faster reading after first epoch finished. However it highly depends on data size and the os/build. 
