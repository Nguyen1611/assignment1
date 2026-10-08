*Qwen2.5-7B original vs primary EGSC (retry on execution error/timeout, up to 2 retries): EX/EM overall and per difficulty. unparseable = predictions the official parser cannot parse (EM=0); em_parseable = EM on the parseable subset; ex_unparseable = EX on the unparseable subset.*

| system       |   ex_easy |   em_easy |   ex_medium |   em_medium |   ex_hard |   em_hard |   ex_extra |   em_extra |   ex_all |   em_all |    n |
|:-------------|----------:|----------:|------------:|------------:|----------:|----------:|-----------:|-----------:|---------:|---------:|-----:|
| original     |  0.870968 |  0.520161 |    0.804933 |    0.282511 |  0.557471 |  0.224138 |   0.518072 |  0.0662651 | 0.733075 | 0.294971 | 1034 |
| primary EGSC |  0.875    |  0.520161 |    0.809417 |    0.282511 |  0.591954 |  0.235632 |   0.542169 |  0.0722892 | 0.745648 | 0.297872 | 1034 |
