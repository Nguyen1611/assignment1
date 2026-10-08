*Trace length measured by tokenizing the reasoning text (Qwen/Qwen3-32B, tokenizers 0.23.2) versus provider-reported counts. Provider reasoning_tokens is an estimate (DeepInfra) and can exceed completion_tokens, so length_tokens (text) is used in all tests; length_chars is reported as a robustness check.*

|   n |   mean_text_tokens |   mean_provider_reasoning_tokens |   mean_provider_completion_tokens |   mean_chars |   provider_reasoning_gt_completion |   text_tokens_gt_completion |   spearman_text_vs_provider |
|----:|-------------------:|---------------------------------:|----------------------------------:|-------------:|-----------------------------------:|----------------------------:|----------------------------:|
|  10 |              448.3 |                            494.2 |                             473.7 |       1954.8 |                                 10 |                           0 |                    0.987879 |
