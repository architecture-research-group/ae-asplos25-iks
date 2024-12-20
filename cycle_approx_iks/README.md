### Cycle Approximate Simulator 

This is a cycle approximate simulator for IKS. It models the internal structure of IKS, using timings from RTL synthesis as well as directly performing the final stage of aggregation, to approximate the performance of IKS. OpenMP parallelization is used to ensure that the core performing final aggregation does not have partial top-k results in its l1 cache.
 
### Usage

#### Building the simulator:

```bash
make simulator 
```
#### Running the simulator:

```bash
./simulator <vector dimension> <number of mac units> <number of PEs> <number of NMAs> <number of IKS> <corpus size (vectors)> <batch size>
```

#### Generating results:

```bash
make table_3
make figure_9
```

#### Examples:

- 1 IKS, 50g corpus, 32 batch size
```bash 
./simulator 768 68 64 8 1 32552084 32 2>/dev/null
```
- Output:
```
Stall cycles: 0
Useful cycles: 45956352
Top-k time: 191 us
Total time: 46.1474 ms
```
