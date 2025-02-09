# Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing

[ICLR 2025] This is an implementation of *Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing*

![Main Method](asset/method.png)

**Probe Pruning (PP)** is executed in four stages: (1) **PP** selects key samples and tokens from the layer-normalized hidden states, based on residual importance, to create a \textit{small yet crucial} probe. (2) **PP** deploys this probe to run a few model layers ahead and obtains the probe's intermediate hidden states. (3) **PP** integrates the probing states with historical states and uses the integrated states to calculate the pruning metric and prune weight channels. (4) **PP** performs full inference on the remaining weights.

## Requirements

- Requires Python 3.9

- See pprequirements.txt
  
- See localtunedllmrequirements.txt for running LLM-Pruner and LoRA-Prune tuned models

## Instruction

- Global hyperparameters are configured in config.yml

- Hyperparameters can be found at hyper.py in modules


## Examples

- Train CIFAR10 dataset with CNN, 0.1 active rate, 100 clients, K=1, DynamicAvg, Dynamic-0.3, 'a-g' interval

  ```ruby
  python train_classifier_fl.py --control_name CIFAR10_cnn_0.1_100_non-iid-l-1_dynamicfl_5_0.3-0.7_1-0_6-1
  ```

- Train CIFAR100 dataset with ResNet-18, 0.1 active rate, 100 clients, Dir(0.3), DynamicAvg, Fix-0.3, 'b-f' interval

  ```ruby
  python train_classifier_fl.py --control_name CIFAR100_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_5_1-0_0.3-0.7_5-2
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

