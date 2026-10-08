*Mean latency and token usage per example, and total cost. Token counts are provider-reported: mean_output_tokens is completion_tokens and mean_reasoning_tokens is the provider's reasoning_tokens, which is an estimate on some providers (DeepInfra) and can exceed completion_tokens; Task 2 measures trace length from the text. Prices: qwen/qwen-2.5-7b-instruct: price not set in config.PRICES (cost shown as NaN); qwen/qwen-2.5-72b-instruct: price not set in config.PRICES (cost shown as NaN); qwen/qwen3-32b: price not set in config.PRICES (cost shown as NaN).*

| model                |    n |   mean_latency_s |   mean_input_tokens |   mean_output_tokens |   mean_reasoning_tokens |   total_cost_usd |
|:---------------------|-----:|-----------------:|--------------------:|---------------------:|------------------------:|-----------------:|
| Qwen2.5-7B-Instruct  | 1034 |         0.572613 |             836.975 |              37.8975 |                    0    |              nan |
| Qwen2.5-72B-Instruct | 1034 |         1.91221  |             836.975 |              34.3714 |                    0    |              nan |
| Qwen3-32B (thinking) | 1034 |        30.9982   |             837.975 |             981.473  |                 1023.61 |              nan |
