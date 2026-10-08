*Trace features by difficulty (count, mean, median, quartiles).*

| feature         | difficulty   |   count |        mean |     median |         q1 |         q3 |        iqr |
|:----------------|:-------------|--------:|------------:|-----------:|-----------:|-----------:|-----------:|
| length_tokens   | easy         |     248 |  507.44     |  362       |  278.5     |  554.5     |  276       |
| length_tokens   | extra        |     166 | 1594.19     | 1267.5     |  850.25    | 1990.75    | 1140.5     |
| length_tokens   | hard         |     174 | 1396.18     | 1073       |  637.5     | 1901.5     | 1264       |
| length_tokens   | medium       |     446 |  752.283    |  503.5     |  381.25    |  804.75    |  423.5     |
| length_chars    | easy         |     248 | 2270.23     | 1629       | 1262       | 2432.75    | 1170.75    |
| length_chars    | extra        |     166 | 6768.25     | 5496       | 3678.75    | 8741.75    | 5063       |
| length_chars    | hard         |     174 | 5999.82     | 4432       | 2714.75    | 8004       | 5289.25    |
| length_chars    | medium       |     446 | 3247.93     | 2240       | 1705.25    | 3477       | 1771.75    |
| marker_count    | easy         |     248 |    3.58468  |    3       |    2       |    5       |    3       |
| marker_count    | extra        |     166 |    8.78313  |    8       |    5       |   12       |    7       |
| marker_count    | hard         |     174 |    7.87931  |    7       |    4       |   10       |    6       |
| marker_count    | medium       |     446 |    4.91704  |    4       |    2       |    6       |    4       |
| markers_per_1k  | easy         |     248 |    7.1682   |    7.13861 |    5.00071 |    9.04977 |    4.04906 |
| markers_per_1k  | extra        |     166 |    5.93755  |    5.66493 |    4.427   |    7.25954 |    2.83254 |
| markers_per_1k  | hard         |     174 |    6.3051   |    6.05969 |    4.80295 |    7.7627  |    2.95975 |
| markers_per_1k  | medium       |     446 |    7.1363   |    6.94847 |    5.11715 |    9.00901 |    3.89186 |
| sql_drafts      | easy         |     248 |    1.55645  |    1       |    1       |    2       |    1       |
| sql_drafts      | extra        |     166 |    6.28313  |    4       |    2       |    7       |    5       |
| sql_drafts      | hard         |     174 |    6.58046  |    4       |    2       |    8       |    6       |
| sql_drafts      | medium       |     446 |    2.17713  |    1       |    1       |    2       |    1       |
| table_coverage  | easy         |     248 |    1        |    1       |    1       |    1       |    0       |
| table_coverage  | extra        |     166 |    0.997992 |    1       |    1       |    1       |    0       |
| table_coverage  | hard         |     174 |    1        |    1       |    1       |    1       |    0       |
| table_coverage  | medium       |     446 |    1        |    1       |    1       |    1       |    0       |
| column_coverage | easy         |     206 |    0.98665  |    1       |    1       |    1       |    0       |
| column_coverage | extra        |     166 |    0.981928 |    1       |    1       |    1       |    0       |
| column_coverage | hard         |     174 |    0.98659  |    1       |    1       |    1       |    0       |
| column_coverage | medium       |     446 |    0.995703 |    1       |    1       |    1       |    0       |
