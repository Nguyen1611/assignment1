*Spearman correlation with ordinal difficulty (easy=1..extra=4); Holm-adjusted.*

| feature         |    n |   n_dropped |        rho |       p_raw |      p_holm |
|:----------------|-----:|------------:|-----------:|------------:|------------:|
| length_tokens   | 1034 |           0 |  0.559686  | 2.72091e-86 | 1.90464e-85 |
| length_chars    | 1034 |           0 |  0.552236  | 1.34172e-83 | 8.05029e-83 |
| marker_count    | 1034 |           0 |  0.454562  | 7.35164e-54 | 3.67582e-53 |
| markers_per_1k  | 1034 |           0 | -0.155709  | 4.86323e-07 | 1.45897e-06 |
| sql_drafts      | 1034 |           0 |  0.440496  | 2.55902e-50 | 1.02361e-49 |
| table_coverage  | 1034 |           0 | -0.0477641 | 0.124805    | 0.124805    |
| column_coverage |  992 |          42 | -0.0876952 | 0.00571162  | 0.0114232   |
