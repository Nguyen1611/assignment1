*Trace features by EX correctness (count, mean, median, quartiles).*

| feature         | correct   |   count |        mean |     median |         q1 |         q3 |        iqr |
|:----------------|:----------|--------:|------------:|-----------:|-----------:|-----------:|-----------:|
| length_tokens   | correct   |     809 |  778.77     |  517       |  356       |  885       |  529       |
| length_tokens   | incorrect |     225 | 1506.26     | 1189       |  696       | 1929       | 1233       |
| length_chars    | correct   |     809 | 3367.5      | 2271       | 1586       | 3912       | 2326       |
| length_chars    | incorrect |     225 | 6465.7      | 5046       | 3186       | 8361       | 5175       |
| marker_count    | correct   |     809 |    4.95921  |    4       |    2       |    7       |    5       |
| marker_count    | incorrect |     225 |    8.44     |    8       |    5       |   11       |    6       |
| markers_per_1k  | correct   |     809 |    6.90683  |    6.62252 |    4.97288 |    8.65801 |    3.68513 |
| markers_per_1k  | incorrect |     225 |    6.46931  |    6.24756 |    4.46429 |    7.88782 |    3.42353 |
| sql_drafts      | correct   |     809 |    2.90729  |    1       |    1       |    3       |    2       |
| sql_drafts      | incorrect |     225 |    5.30222  |    3       |    1       |    5       |    4       |
| table_coverage  | correct   |     809 |    1        |    1       |    1       |    1       |    0       |
| table_coverage  | incorrect |     225 |    0.998519 |    1       |    1       |    1       |    0       |
| column_coverage | correct   |     769 |    0.995709 |    1       |    1       |    1       |    0       |
| column_coverage | incorrect |     223 |    0.969955 |    1       |    1       |    1       |    0       |
