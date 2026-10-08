*Mann-Whitney U, correct vs incorrect, with rank-biserial effect size; Holm-adjusted p-values across features.*

| feature         |   n_correct |   n_incorrect |   n_dropped |        U |   rank_biserial |       p_raw |      p_holm |
|:----------------|------------:|--------------:|------------:|---------:|----------------:|------------:|------------:|
| length_tokens   |         809 |           225 |           0 |  45442.5 |     -0.5007     | 1.30625e-30 | 7.83748e-30 |
| length_chars    |         809 |           225 |           0 |  45057   |     -0.504936   | 4.21201e-31 | 2.94841e-30 |
| marker_count    |         809 |           225 |           0 |  49568   |     -0.455372   | 7.14399e-26 | 3.57199e-25 |
| markers_per_1k  |         809 |           225 |           0 | 100117   |      0.100036   | 0.0215805   | 0.043161    |
| sql_drafts      |         809 |           225 |           0 |  62934.5 |     -0.308507   | 3.36536e-14 | 1.34614e-13 |
| table_coverage  |         809 |           225 |           0 |  91417   |      0.00444444 | 0.0582448   | 0.0582448   |
| column_coverage |         769 |           223 |          42 |  90124   |      0.0510884  | 5.06159e-05 | 0.000151848 |
