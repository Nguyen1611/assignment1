*Pairwise exact McNemar tests on per-example EX (same 1,034 examples); Holm-adjusted across the three pairs.*

| model_a              | model_b              |     ex_a |     ex_b |   only_a_correct |   only_b_correct |   mcnemar_p |      p_holm |
|:---------------------|:---------------------|---------:|---------:|-----------------:|-----------------:|------------:|------------:|
| Qwen2.5-7B-Instruct  | Qwen2.5-72B-Instruct | 0.733075 | 0.811412 |               44 |              125 | 3.47445e-10 | 1.04234e-09 |
| Qwen2.5-7B-Instruct  | Qwen3-32B (thinking) | 0.733075 | 0.782398 |               68 |              119 | 0.000235943 | 0.000471886 |
| Qwen2.5-72B-Instruct | Qwen3-32B (thinking) | 0.811412 | 0.782398 |               82 |               52 | 0.0119515   | 0.0119515   |
