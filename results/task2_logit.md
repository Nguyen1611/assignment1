*Logistic regression (maximum likelihood): EX on z-scored trace features plus difficulty (reference: easy), n=992. table_coverage is omitted (1.0 for all but one trace, causing quasi-separation); length_chars is omitted (collinear with length_tokens).*

| term                                       |      coef |        se |        z |           p |   odds_ratio |   or_ci_low |   or_ci_high |   n |
|:-------------------------------------------|----------:|----------:|---------:|------------:|-------------:|------------:|-------------:|----:|
| Intercept                                  |  1.95156  | 0.234604  |  8.3185  | 8.90835e-17 |     7.03963  |    4.44482  |    11.1492   | 992 |
| C(difficulty, Treatment('easy'))[T.medium] | -0.420867 | 0.267767  | -1.57177 | 0.116005    |     0.656477 |    0.388415 |     1.10954  | 992 |
| C(difficulty, Treatment('easy'))[T.hard]   | -0.792116 | 0.305125  | -2.59603 | 0.00943066  |     0.452885 |    0.249038 |     0.823591 | 992 |
| C(difficulty, Treatment('easy'))[T.extra]  | -1.25027  | 0.299198  | -4.17875 | 2.93121e-05 |     0.286426 |    0.159344 |     0.514862 | 992 |
| z_length_tokens                            | -0.92726  | 0.129216  | -7.17603 | 7.17645e-13 |     0.395636 |    0.307119 |     0.509666 | 992 |
| z_markers_per_1k                           | -0.153362 | 0.0932258 | -1.64506 | 0.0999576   |     0.857819 |    0.714565 |     1.02979  | 992 |
| z_sql_drafts                               |  0.40875  | 0.118641  |  3.44526 | 0.000570507 |     1.50493  |    1.1927   |     1.89891  | 992 |
| z_column_coverage                          |  0.326578 | 0.0846258 |  3.85909 | 0.00011381  |     1.38622  |    1.17435  |     1.63631  | 992 |
