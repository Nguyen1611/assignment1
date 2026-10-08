*Experimental environment: local machine (orchestration and evaluation only), packages, and remote inference providers.*

| item                                     | value                                                                            |
|:-----------------------------------------|:---------------------------------------------------------------------------------|
| OS                                       | Linux                                                                            |
| Kernel                                   | 6.18.40.1-microsoft-standard-WSL2                                                |
| Running under WSL                        | True                                                                             |
| Distribution                             | Ubuntu 24.04.1 LTS                                                               |
| CPU                                      | 13th Gen Intel(R) Core(TM) i7-13620H                                             |
| Logical cores                            | 16                                                                               |
| RAM (GiB)                                | 7.61                                                                             |
| RAM note                                 | RAM allocated to the WSL VM (from /proc/meminfo), not total host RAM             |
| Python                                   | 3.12.3                                                                           |
| nltk version                             | 3.10.3                                                                           |
| openai version                           | 3.26.0                                                                           |
| pandas version                           | 3.0.6                                                                            |
| scipy version                            | 1.18.1                                                                           |
| sqlparse version                         | 0.6.0                                                                            |
| statsmodels version                      | 0.15.0                                                                           |
| Official script commit                   | e97acc546ecbee8fa27fa8dbf025ef61493a876c                                         |
| Inference                                | api via OpenRouter on remote provider hardware not controlled by the student     |
| Local machine role                       | orchestration, SQL execution, evaluation                                         |
| Providers for qwen/qwen-2.5-72b-instruct | DeepInfra (7 calls, quant: fp8), Novita (3 calls, quant: bf16)                   |
| Providers for qwen/qwen-2.5-7b-instruct  | Phala (10 calls, quant: unknown)                                                 |
| Providers for qwen/qwen3-32b             | DeepInfra (10 calls, quant: fp8)                                                 |
| Run 00_setup_data:download               | 2026-10-07T22:42:24.411020-04:00 -> 2026-10-07T22:42:24.412443-04:00 (completed) |
| Run 00_setup_data:gold_vs_gold           | 2026-10-07T22:43:10.990567-04:00 -> 2026-10-07T22:43:12.647875-04:00 (completed) |
| Run 00_setup_data:wrapper_vs_cli         | 2026-10-07T22:43:29.957504-04:00 -> 2026-10-07T22:43:32.284799-04:00 (completed) |
| Run qwen25_7b                            | 2026-10-07T22:45:39.284235-04:00 -> 2026-10-07T22:45:45.916094-04:00 (completed) |
| Run qwen25_72b                           | 2026-10-07T22:50:31.872974-04:00 -> 2026-10-07T22:50:38.308828-04:00 (completed) |
| Run qwen3_32b_thinking:preflight         | 2026-10-07T22:55:45.051532-04:00 -> 2026-10-07T22:56:25.011359-04:00 (completed) |
| Run qwen3_32b_thinking                   | 2026-10-07T22:56:39.019166-04:00 -> 2026-10-07T22:57:19.585463-04:00 (completed) |
