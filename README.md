# HW-GPT-Bench
## Repository for HW-GPT Benchmark
![alt text](figures/overview.png)

## To install in editable mode (-e) run:

```sh
$ git clone https://github.com/automl/HW-Aware-LLM-Bench
$ cd HW-Aware-LLM-Bench
$ pip install -e .
```

## Example api usage
```python
from hwgpt.api import HWGPT
api = HWGPT(search_space="s",use_supernet_surrogate=False) # initialize API
random_arch = api.sample_arch() # sample random arch
api.set_arch(random_arch) # set  arch
results = api.query() # query all for the sampled arch
print("Results: ", results)
energy = api.query(metric="energies") # query energy
print("Energy: ", energy)
rtx2080 = api.query(device="rtx2080") # query device
print("RTX2080: ", rtx2080)
# query perplexity based on mlp predictor
perplexity_mlp = api.query(metric="perplexity",predictor="mlp")
print("Perplexity MLP: ", perplexity_mlp)
```