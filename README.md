# Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing

[ICLR 2025] This is an implementation of *Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing*

![Main Method](asset/method.png)

**Probe Pruning (PP)** is executed in four stages: (1) **PP** selects key samples and tokens from the layer-normalized hidden states, based on residual importance, to create a \textit{small yet crucial} probe. (2) **PP** deploys this probe to run a few model layers ahead and obtains the probe's intermediate hidden states. (3) **PP** integrates the probing states with historical states and uses the integrated states to calculate the pruning metric and prune weight channels. (4) **PP** performs full inference on the remaining weights.

## Requirements

- Requires Python 3.9.
- See pprequirements.txt.
- See localtunedllmrequirements.txt for running LLM-Pruner and LoRA-Prune tuned models.
- C4 calibration dataset can be found at [here](https://drive.google.com/file/d/1dTl7rPeOqKqQmFPxldITolJTVAp8MScv/view?usp=sharing). Please download it and place it under data/c4.

## Instruction

- Global hyperparameters are configured in config.yml.

- Hyperparameters can be found at hyper.py in modules.

- Code for tuning model using LLM-Pruner and LoRA-Prune is available [here](https://github.com/wangziyannb/finetuning-baseline-of-PP) and example tuned models are available [here](https://drive.google.com/drive/folders/170q3BcGBaG8mGu7KTc6eVxrN36S-dNNJ).


## Examples

- Test WikiText2 dataset using Probe Pruning with the default probe at a 40% pruning ratio on LLaMA-2-7B.

  ```ruby
  python test_model.py --control_name wikitext-2v1_llama-2-7b_clm_20_1024_0.4_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default --device cuda
  ```

- Test WikiText2 dataset using FLAP at a 40% pruning ratio on LLaMA-2-7B.

  ```ruby
  python test_model.py --control_name wikitext-2v1_llama-2-7b_clm_20_1024_0.4_flap_flap-default_asyncinter_c4-2000_None_default --device cuda
  ```


## Results

*Zero-shot Performance of LLaMA-2-7B/13B and OPT-13B After Pruning Attention and MLP Blocks Without Fine-Tuning, PP demonstrates superior performance in nearly all scenarios.*

| Method                  | Pruning Ratio | LLaMA-2-7B (Text Generation) ↓ | LLaMA-2-13B (Text Generation) ↓ | OPT-13B (Text Generation) ↓ | LLaMA-2-7B (Commonsense Reasoning) ↑ | LLaMA-2-13B (Commonsense Reasoning) ↑ | OPT-13B (Commonsense Reasoning) ↑ |
| ----------------------- | ------------- | ------------------------------ | ------------------------------- | --------------------------- | ------------------------------------ | ------------------------------------- | --------------------------------- |
| **Dense**               | 0%            | 6.0 (0.1)                      | 5.1 (0.1)                       | 11.6 (0.1)                  | 64.0                                 | 66.2                                  | 57.2                              |
| **Full-Batch Probing**  | 20%           | 7.3 (0.1)                      | 6.2 (0.1)                       | 12.6 (0.1)                  | 62.6                                 | 65.3                                  | 56.4                              |
| **Wanda-sp**            | 20%           | 10.6 (0.1)                     | 9.0 (0.1)                       | 17.4 (0.1)                  | 61.5                                 | 65.0                                  | 55.2                              |
| **FLAP**                | 20%           | 10.3 (0.1)                     | 7.5 (0.1)                       | 18.8 (0.2)                  | 61.4                                 | 64.6                                  | 54.9                              |
| **LoRAPrune w/o LoRA**  | 20%           | 22.7 (0.9)                     | 16.1 (0.7)                      | N/A                         | 57.9                                 | 58.9                                  | N/A                               |
| **LLM-Pruner w/o LoRA** | 20%           | 17.5 (1.6)                     | 11.3 (0.7)                      | N/A                         | 57.4                                 | 61.3                                  | N/A                               |
| **PP**                  | 20%           | **8.1 (0.1)**                  | **6.7 (0.1)**                   | **14.7 (0.1)**              | **62.8**                             | **65.3**                              | **56.5**                          |
| **Full-Batch Probing**  | 40%           | 13.6 (0.1)                     | 8.9 (0.1)                       | 17.9 (0.2)                  | 58.7                                 | 62.9                                  | 54.0                              |
| **Wanda-sp**            | 40%           | 43.8 (1.5)                     | 21.6 (0.4)                      | 42.7 (0.7)                  | 54.8                                 | 56.6                                  | 50.5                              |
| **FLAP**                | 40%           | 38.9 (1.3)                     | 15.5 (0.0)                      | 51.0 (0.7)                  | 54.9                                 | 60.6                                  | 50.8                              |
| **LoRAPrune w/o LoRA**  | 40%           | 129.5 (3.0)                    | 74.8 (6.4)                      | N/A                         | 45.4                                 | 48.1                                  | N/A                               |
| **LLM-Pruner w/o LoRA** | 40%           | 51.1 (4.3)                     | 34.5 (2.4)                      | N/A                         | 47.8                                 | 52.0                                  | N/A                               |
| **PP**                  | 40%           | **16.8 (0.1)**                 | **11.3 (0.1)**                  | **26.7 (0.3)**              | **56.6**                             | **61.0**                              | **53.1**                          |

