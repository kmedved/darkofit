| dataset | weight_mode | model | seed | status | fit_s | rmse_mu | nll | crps | cov90 | width90 | cov90_by_sigma | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 0 | ok | 0.639 | 0.74921 | 1.02727 | 0.40190 | 0.900 | 2.360 | 0.901/0.897/0.899/0.894/0.911 |  |
| synthetic_100k | none | darkofit_gaussian_es_conformal | 0 | ok | 0.652 | 0.74921 | 1.02744 | 0.40192 | 0.896 | 2.325 | 0.900/0.895/0.896/0.889/0.903 |  |
| synthetic_100k | none | ngboost | 0 | ok | 31.585 | 0.73658 | 1.00189 | 0.39435 | 0.897 | 2.277 | 0.910/0.899/0.896/0.893/0.889 |  |
| synthetic_100k | none | catboost_uncertainty | 0 | ok | 0.408 | 0.74653 | 1.02465 | 0.40068 | 0.901 | 2.351 | 0.910/0.900/0.898/0.895/0.904 |  |
| synthetic_100k | none | lightgbm_quantile_pair | 0 | ok | 0.524 | - | - | - | 0.899 | 2.404 | - | interval-only baseline |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 1 | ok | 0.664 | 0.74174 | 1.01675 | 0.39793 | 0.901 | 2.333 | 0.902/0.899/0.900/0.905/0.900 |  |
| synthetic_100k | none | darkofit_gaussian_es_conformal | 1 | ok | 0.701 | 0.74174 | 1.01677 | 0.39793 | 0.900 | 2.330 | 0.900/0.897/0.899/0.905/0.900 |  |
| synthetic_100k | none | ngboost | 1 | ok | 31.464 | 0.72870 | 0.99004 | 0.39000 | 0.899 | 2.266 | 0.914/0.903/0.893/0.905/0.882 |  |
| synthetic_100k | none | catboost_uncertainty | 1 | ok | 0.402 | 0.73869 | 1.01296 | 0.39643 | 0.904 | 2.335 | 0.916/0.904/0.899/0.905/0.897 |  |
| synthetic_100k | none | lightgbm_quantile_pair | 1 | ok | 0.450 | - | - | - | 0.902 | 2.394 | - | interval-only baseline |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 2 | ok | 0.600 | 0.74270 | 1.01899 | 0.39893 | 0.899 | 2.324 | 0.901/0.894/0.900/0.897/0.904 |  |
| synthetic_100k | none | darkofit_gaussian_es_conformal | 2 | ok | 0.601 | 0.74270 | 1.01902 | 0.39893 | 0.902 | 2.346 | 0.903/0.896/0.903/0.901/0.908 |  |
| synthetic_100k | none | ngboost | 2 | ok | 30.479 | 0.73094 | 0.99732 | 0.39222 | 0.897 | 2.257 | 0.905/0.894/0.898/0.895/0.891 |  |
| synthetic_100k | none | catboost_uncertainty | 2 | ok | 0.367 | 0.74059 | 1.01820 | 0.39832 | 0.902 | 2.332 | 0.910/0.905/0.900/0.895/0.900 |  |
| synthetic_100k | none | lightgbm_quantile_pair | 2 | ok | 0.475 | - | - | - | 0.899 | 2.391 | - | interval-only baseline |
| synthetic_t3_100k | none | darkofit_gaussian_es_calibrated | 0 | ok | 0.608 | 0.72920 | 1.02968 | 0.34893 | 0.925 | 2.232 | 0.924/0.919/0.929/0.926/0.927 |  |
| synthetic_t3_100k | none | darkofit_gaussian_es_conformal | 0 | ok | 0.561 | 0.72920 | 1.02769 | 0.34941 | 0.893 | 1.898 | 0.904/0.890/0.894/0.886/0.892 |  |
| synthetic_t3_100k | none | ngboost | 0 | ok | 30.607 | 0.71578 | 1.00977 | 0.33919 | 0.922 | 2.137 | 0.921/0.925/0.920/0.924/0.919 |  |
| synthetic_t3_100k | none | catboost_uncertainty | 0 | ok | 0.395 | 0.72934 | 1.01888 | 0.35078 | 0.930 | 2.289 | 0.931/0.930/0.924/0.931/0.934 |  |
| synthetic_t3_100k | none | lightgbm_quantile_pair | 0 | ok | 0.483 | - | - | - | 0.900 | 2.027 | - | interval-only baseline |
| synthetic_t3_100k | none | darkofit_gaussian_es_calibrated | 1 | ok | 0.563 | 0.73354 | 1.00441 | 0.35433 | 0.931 | 2.310 | 0.936/0.931/0.934/0.930/0.925 |  |
| synthetic_t3_100k | none | darkofit_gaussian_es_conformal | 1 | ok | 0.559 | 0.73354 | 1.00401 | 0.35412 | 0.905 | 2.005 | 0.909/0.907/0.909/0.902/0.898 |  |
| synthetic_t3_100k | none | ngboost | 1 | ok | 31.383 | 0.71836 | 0.97996 | 0.39922 | 0.918 | 2.906 | 0.918/0.923/0.922/0.920/0.908 |  |
| synthetic_t3_100k | none | catboost_uncertainty | 1 | ok | 0.484 | 0.73240 | 1.00674 | 0.35338 | 0.925 | 2.239 | 0.931/0.924/0.928/0.923/0.921 |  |
| synthetic_t3_100k | none | lightgbm_quantile_pair | 1 | ok | 0.564 | - | - | - | 0.898 | 2.014 | - | interval-only baseline |
| synthetic_t3_100k | none | darkofit_gaussian_es_calibrated | 2 | ok | 0.602 | 0.72611 | 1.00857 | 0.35179 | 0.931 | 2.320 | 0.933/0.934/0.929/0.923/0.934 |  |
| synthetic_t3_100k | none | darkofit_gaussian_es_conformal | 2 | ok | 0.639 | 0.72611 | 1.00859 | 0.35065 | 0.898 | 1.956 | 0.898/0.900/0.895/0.891/0.908 |  |
| synthetic_t3_100k | none | ngboost | 2 | ok | 31.483 | 0.71156 | 0.98217 | 0.34267 | 0.919 | 2.157 | 0.921/0.920/0.915/0.920/0.919 |  |
| synthetic_t3_100k | none | catboost_uncertainty | 2 | ok | 0.367 | 0.72944 | 1.01792 | 0.35359 | 0.927 | 2.268 | 0.933/0.930/0.920/0.920/0.929 |  |
| synthetic_t3_100k | none | lightgbm_quantile_pair | 2 | ok | 0.505 | - | - | - | 0.898 | 2.016 | - | interval-only baseline |
| openml_cpu_act | none | darkofit_gaussian_es_calibrated | 0 | ok | 0.232 | 2.48035 | 2.22010 | 1.23800 | 0.909 | 7.421 | 0.956/0.890/0.910/0.892/0.895 |  |
| openml_cpu_act | none | darkofit_gaussian_es_conformal | 0 | ok | 0.241 | 2.48035 | 2.21836 | 1.23968 | 0.894 | 6.989 | 0.944/0.873/0.895/0.868/0.888 |  |
| openml_cpu_act | none | ngboost | 0 | ok | 3.402 | 2.48148 | 2.33166 | 1.23988 | 0.886 | 6.481 | 0.922/0.907/0.915/0.851/0.836 |  |
| openml_cpu_act | none | catboost_uncertainty | 0 | ok | 0.203 | 2.54183 | 2.20891 | 1.30372 | 0.910 | 7.562 | 0.944/0.924/0.920/0.863/0.897 |  |
| openml_cpu_act | none | lightgbm_quantile_pair | 0 | ok | 0.262 | - | - | - | 0.838 | 10.878 | - | interval-only baseline |
| openml_cpu_act | none | darkofit_gaussian_es_calibrated | 1 | ok | 0.250 | 2.57886 | 2.26712 | 1.24622 | 0.918 | 7.502 | 0.959/0.941/0.924/0.868/0.897 |  |
| openml_cpu_act | none | darkofit_gaussian_es_conformal | 1 | ok | 0.270 | 2.58063 | 2.26645 | 1.24966 | 0.906 | 7.354 | 0.949/0.922/0.912/0.851/0.895 |  |
| openml_cpu_act | none | ngboost | 1 | ok | 3.611 | 2.38443 | 2.36000 | 1.21796 | 0.886 | 6.423 | 0.905/0.922/0.871/0.873/0.858 |  |
| openml_cpu_act | none | catboost_uncertainty | 1 | ok | 0.302 | 2.92273 | 2.26127 | 1.38448 | 0.907 | 7.644 | 0.929/0.932/0.893/0.897/0.883 |  |
| openml_cpu_act | none | lightgbm_quantile_pair | 1 | ok | 0.242 | - | - | - | 0.834 | 12.091 | - | interval-only baseline |
| openml_cpu_act | none | darkofit_gaussian_es_calibrated | 2 | ok | 0.279 | 2.57255 | 2.26351 | 1.36693 | 0.938 | 9.884 | 0.978/0.971/0.934/0.897/0.907 |  |
| openml_cpu_act | none | darkofit_gaussian_es_conformal | 2 | ok | 0.254 | 2.53920 | 2.23158 | 1.31438 | 0.926 | 8.493 | 0.983/0.961/0.932/0.858/0.897 |  |
| openml_cpu_act | none | ngboost | 2 | ok | 3.552 | 2.28195 | 2.12659 | 1.20153 | 0.886 | 6.456 | 0.924/0.893/0.885/0.868/0.861 |  |
| openml_cpu_act | none | catboost_uncertainty | 2 | ok | 0.187 | 2.57116 | 2.21955 | 1.32792 | 0.917 | 7.638 | 0.934/0.941/0.910/0.892/0.907 |  |
| openml_cpu_act | none | lightgbm_quantile_pair | 2 | ok | 0.224 | - | - | - | 0.833 | 10.775 | - | interval-only baseline |
| openml_wine_quality | none | darkofit_gaussian_es_calibrated | 0 | ok | 0.165 | 0.68762 | 1.02056 | 0.37804 | 0.903 | 2.189 | 0.880/0.908/0.920/0.902/0.905 |  |
| openml_wine_quality | none | darkofit_gaussian_es_conformal | 0 | ok | 0.161 | 0.68684 | 1.02847 | 0.37757 | 0.907 | 2.253 | 0.880/0.917/0.926/0.902/0.911 |  |
| openml_wine_quality | none | ngboost | 0 | ok | 1.554 | 0.69602 | 1.05783 | 0.38590 | 0.866 | 2.011 | 0.818/0.883/0.868/0.902/0.862 |  |
| openml_wine_quality | none | catboost_uncertainty | 0 | ok | 0.100 | 0.69680 | 1.04348 | 0.38425 | 0.875 | 2.013 | 0.877/0.889/0.883/0.892/0.834 |  |
| openml_wine_quality | none | lightgbm_quantile_pair | 0 | ok | 0.172 | - | - | - | 0.924 | 2.243 | - | interval-only baseline |
| openml_wine_quality | none | darkofit_gaussian_es_calibrated | 1 | ok | 0.163 | 0.67382 | 1.00572 | 0.37234 | 0.913 | 2.227 | 0.929/0.908/0.886/0.911/0.929 |  |
| openml_wine_quality | none | darkofit_gaussian_es_conformal | 1 | ok | 0.154 | 0.67059 | 0.99953 | 0.37031 | 0.924 | 2.296 | 0.935/0.902/0.914/0.926/0.945 |  |
| openml_wine_quality | none | ngboost | 1 | ok | 1.541 | 0.68403 | 1.02101 | 0.37857 | 0.878 | 2.039 | 0.886/0.895/0.868/0.846/0.895 |  |
| openml_wine_quality | none | catboost_uncertainty | 1 | ok | 0.098 | 0.67903 | 1.00876 | 0.37415 | 0.890 | 2.033 | 0.935/0.895/0.874/0.871/0.877 |  |
| openml_wine_quality | none | lightgbm_quantile_pair | 1 | ok | 0.185 | - | - | - | 0.924 | 2.165 | - | interval-only baseline |
| openml_wine_quality | none | darkofit_gaussian_es_calibrated | 2 | ok | 0.155 | 0.69001 | 1.02827 | 0.37966 | 0.887 | 2.077 | 0.914/0.932/0.846/0.880/0.862 |  |
| openml_wine_quality | none | darkofit_gaussian_es_conformal | 2 | ok | 0.170 | 0.68808 | 1.03195 | 0.37849 | 0.900 | 2.162 | 0.911/0.935/0.874/0.886/0.892 |  |
| openml_wine_quality | none | ngboost | 2 | ok | 1.601 | 0.70134 | 1.04978 | 0.38754 | 0.873 | 2.019 | 0.892/0.883/0.868/0.883/0.840 |  |
| openml_wine_quality | none | catboost_uncertainty | 2 | ok | 0.090 | 0.69849 | 1.04328 | 0.38478 | 0.874 | 2.008 | 0.895/0.917/0.877/0.852/0.831 |  |
| openml_wine_quality | none | lightgbm_quantile_pair | 2 | ok | 0.171 | - | - | - | 0.922 | 2.324 | - | interval-only baseline |
| openml_boston | none | darkofit_gaussian_es_calibrated | 0 | ok | 0.160 | 4.74656 | 3.37072 | 2.18358 | 0.843 | 8.842 | 0.885/0.923/0.760/0.720/0.920 |  |
| openml_boston | none | darkofit_gaussian_es_conformal | 0 | ok | 0.158 | 4.74424 | 3.52912 | 2.18500 | 0.890 | 11.911 | 0.923/1.000/0.800/0.760/0.960 |  |
| openml_boston | none | ngboost | 0 | ok | 0.288 | 3.76829 | 5.75801 | 2.00803 | 0.535 | 3.885 | 0.423/0.500/0.640/0.600/0.520 |  |
| openml_boston | none | catboost_uncertainty | 0 | ok | 0.079 | 4.59243 | 4.29535 | 2.18165 | 0.654 | 5.441 | 0.654/0.538/0.760/0.840/0.480 |  |
| openml_boston | none | lightgbm_quantile_pair | 0 | ok | 0.160 | - | - | - | 0.717 | 8.315 | - | interval-only baseline |
| openml_boston | none | darkofit_gaussian_es_calibrated | 1 | ok | 0.159 | 3.77467 | 2.55943 | 1.90183 | 0.913 | 11.213 | 0.962/0.962/0.760/1.000/0.880 |  |
| openml_boston | none | darkofit_gaussian_es_conformal | 1 | ok | 0.165 | 3.72767 | 2.56194 | 1.88149 | 0.929 | 11.737 | 0.962/0.962/0.800/1.000/0.920 |  |
| openml_boston | none | ngboost | 1 | ok | 0.302 | 2.62082 | 3.94959 | 1.59102 | 0.591 | 4.070 | 0.462/0.500/0.560/0.800/0.640 |  |
| openml_boston | none | catboost_uncertainty | 1 | ok | 0.072 | 3.09480 | 2.69439 | 1.64178 | 0.756 | 6.069 | 0.808/0.692/0.880/0.680/0.720 |  |
| openml_boston | none | lightgbm_quantile_pair | 1 | ok | 0.155 | - | - | - | 0.701 | 9.445 | - | interval-only baseline |
| openml_boston | none | darkofit_gaussian_es_calibrated | 2 | ok | 0.137 | 4.44305 | 2.98047 | 2.35775 | 0.984 | 20.629 | 1.000/0.962/0.960/1.000/1.000 |  |
| openml_boston | none | darkofit_gaussian_es_conformal | 2 | ok | 0.137 | 3.91999 | 2.76883 | 1.89213 | 0.984 | 22.435 | 1.000/1.000/0.920/1.000/1.000 |  |
| openml_boston | none | ngboost | 2 | ok | 0.288 | 2.99884 | 4.66670 | 1.71035 | 0.528 | 3.892 | 0.269/0.538/0.520/0.720/0.600 |  |
| openml_boston | none | catboost_uncertainty | 2 | ok | 0.072 | 3.13782 | 2.56425 | 1.49004 | 0.764 | 6.149 | 0.808/0.769/0.720/0.880/0.640 |  |
| openml_boston | none | lightgbm_quantile_pair | 2 | ok | 0.199 | - | - | - | 0.787 | 9.391 | - | interval-only baseline |
