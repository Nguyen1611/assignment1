*Logistic regression (L1-regularized, alpha=0.01; SEs/p-values not valid): EX on z-scored trace features plus difficulty (reference: easy), n=8.*

| term                                       |    coef |       se |          z |          p |   odds_ratio |     or_ci_low |    or_ci_high |   n |
|:-------------------------------------------|--------:|---------:|-----------:|-----------:|-------------:|--------------:|--------------:|----:|
| Intercept                                  | 6.68336 |  10.0063 |   0.667918 |   0.504186 |      799.001 |   2.42773e-06 |   2.62963e+11 |   8 |
| C(difficulty, Treatment('easy'))[T.medium] | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
| z_length_tokens                            | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
| z_markers_per_1k                           | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
| z_sql_drafts                               | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
| z_table_coverage                           | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
| z_column_coverage                          | 0       | nan      | nan        | nan        |        1     | nan           | nan           |   8 |
