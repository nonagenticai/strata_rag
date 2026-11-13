chunks=997194 dim=256 tenants=300+big(2000,8000,25000,60000) load=28s global_hnsw_build=221s hnsw=1.36 GB db=2.95 GB partial_platform=200s/1.16 GB partial_tenant=26s/0.20 GB

| set | n | chunks in set (median) | path | plan | p50 ms | p95 ms | recall@10 mean | recall@10 min | recall@50 mean | short (rows < min(50, set)) |
|---|---|---|---|---|---|---|---|---|---|---|
| tenant_scope | 100 | 61 | exact | store_idx:100 | 1.9 | 6.5 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 61 | V1_mst_default | hnsw:100 | 110.1 | 401.1 | 0.661 | 0.000 | 0.375 | 82 |
| tenant_scope | 100 | 61 | V1_mst_100k | hnsw:100 | 123.5 | 328.8 | 0.683 | 0.100 | 0.391 | 80 |
| tenant_scope | 100 | 61 | V1_mst_1m | hnsw:100 | 78.8 | 149.3 | 0.683 | 0.100 | 0.391 | 80 |
| tenant_scope | 100 | 61 | V1forced_mst_default | hnsw:100 | 42.1 | 65.6 | 0.661 | 0.000 | 0.375 | 82 |
| tenant_scope | 100 | 61 | V1forced_mst_100k | hnsw:100 | 83.6 | 145.6 | 0.683 | 0.100 | 0.391 | 80 |
| tenant_scope | 100 | 61 | V1forced_mst_1m | hnsw:100 | 77.4 | 156.4 | 0.683 | 0.100 | 0.391 | 80 |
| tenant_scope | 100 | 61 | V1_wm16MB | hnsw:100 | 510.9 | 864.1 | 0.713 | 0.100 | 0.446 | 75 |
| tenant_scope | 100 | 61 | V1_wm64MB | hnsw:52,timeout:48 | 4901.7 | 5000.0 | 0.462 | 0.000 | 0.417 | 69 |
| tenant_scope | 100 | 61 | V1_wm256MB | hnsw:23,timeout:77 | 5000.0 | 5000.0 | 0.208 | 0.000 | 0.222 | 77 |
| tenant_scope | 100 | 61 | V2 | store_idx:100 | 0.8 | 3.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 61 | V3 | hnsw_tenant:100 | 41.4 | 59.5 | 0.795 | 0.100 | 0.525 | 74 |
| platform_job | 100 | 342 | exact | store_idx:100 | 1.7 | 5.1 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 342 | V1_mst_default | hnsw:100 | 2.9 | 30.5 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1_mst_100k | hnsw:100 | 0.8 | 2.9 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1_mst_1m | hnsw:100 | 0.8 | 3.0 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1forced_mst_default | hnsw:100 | 0.8 | 3.7 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1forced_mst_100k | hnsw:100 | 0.8 | 2.7 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1forced_mst_1m | hnsw:100 | 0.8 | 2.7 | 0.969 | 0.000 | 0.961 | 1 |
| platform_job | 100 | 342 | V1_wm16MB | hnsw:100 | 1.5 | 9.9 | 0.979 | 0.800 | 0.969 | 0 |
| platform_job | 100 | 342 | V1_wm64MB | hnsw:100 | 1.8 | 11.0 | 0.979 | 0.800 | 0.969 | 0 |
| platform_job | 100 | 342 | V1_wm256MB | hnsw:100 | 1.0 | 4.0 | 0.979 | 0.800 | 0.969 | 0 |
| platform_job | 100 | 342 | V2 | hnsw_platform:100 | 2.2 | 9.4 | 0.955 | 0.600 | 0.959 | 0 |
| platform_job | 100 | 342 | V3 | hnsw_platform:100 | 2.2 | 9.4 | 0.955 | 0.600 | 0.959 | 0 |
| platform_scope | 100 | 53113 | exact | store_idx:100 | 20.6 | 60.1 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 53113 | V1_mst_default | hnsw:100 | 2.5 | 17.6 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1_mst_100k | hnsw:100 | 0.8 | 4.1 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1_mst_1m | hnsw:100 | 0.7 | 4.3 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1forced_mst_default | hnsw:100 | 0.8 | 6.1 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1forced_mst_100k | hnsw:100 | 0.8 | 3.7 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1forced_mst_1m | hnsw:100 | 0.8 | 5.1 | 0.943 | 0.000 | 0.940 | 2 |
| platform_scope | 100 | 53113 | V1_wm16MB | hnsw:100 | 1.2 | 4.9 | 0.963 | 0.400 | 0.960 | 0 |
| platform_scope | 100 | 53113 | V1_wm64MB | hnsw:100 | 1.3 | 5.2 | 0.963 | 0.400 | 0.960 | 0 |
| platform_scope | 100 | 53113 | V1_wm256MB | hnsw:100 | 0.8 | 2.5 | 0.963 | 0.400 | 0.960 | 0 |
| platform_scope | 100 | 53113 | V2 | hnsw_platform:100 | 1.6 | 7.0 | 0.958 | 0.500 | 0.947 | 0 |
| platform_scope | 100 | 53113 | V3 | hnsw_platform:100 | 1.6 | 7.0 | 0.958 | 0.500 | 0.947 | 0 |
| mixed | 100 | 53096 | exact | store_idx:100 | 20.0 | 39.4 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 53096 | V1_mst_default | hnsw:100 | 3.2 | 21.1 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1_mst_100k | hnsw:100 | 0.9 | 3.5 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1_mst_1m | hnsw:100 | 0.9 | 3.2 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1forced_mst_default | hnsw:100 | 1.0 | 4.3 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1forced_mst_100k | hnsw:100 | 0.9 | 3.1 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1forced_mst_1m | hnsw:100 | 0.9 | 3.6 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1_wm16MB | hnsw:100 | 1.7 | 7.9 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1_wm64MB | hnsw:100 | 1.7 | 9.0 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V1_wm256MB | hnsw:100 | 0.9 | 4.3 | 0.943 | 0.400 | 0.949 | 0 |
| mixed | 100 | 53096 | V2 | hnsw_platform+store_idx:100 | 4.1 | 40.0 | 0.934 | 0.100 | 0.901 | 2 |
| mixed | 100 | 53096 | V3 | hnsw_platform+hnsw_tenant:100 | 54.4 | 139.5 | 0.933 | 0.100 | 0.901 | 3 |
| tenant_big_2000 | 50 | 1993 | exact | store_idx:50 | 1.4 | 6.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2000 | 50 | 1993 | V1_mst_default | hnsw:50 | 4.9 | 99.1 | 0.944 | 0.000 | 0.896 | 7 |
| tenant_big_2000 | 50 | 1993 | V1_mst_100k | hnsw:50 | 1.7 | 86.8 | 0.944 | 0.000 | 0.896 | 6 |
| tenant_big_2000 | 50 | 1993 | V1_mst_1m | hnsw:50 | 1.6 | 84.6 | 0.944 | 0.000 | 0.896 | 6 |
| tenant_big_2000 | 50 | 1993 | V1forced_mst_default | hnsw:50 | 1.7 | 44.0 | 0.944 | 0.000 | 0.896 | 7 |
| tenant_big_2000 | 50 | 1993 | V1forced_mst_100k | hnsw:50 | 1.7 | 70.9 | 0.944 | 0.000 | 0.896 | 6 |
| tenant_big_2000 | 50 | 1993 | V1forced_mst_1m | hnsw:50 | 1.7 | 92.3 | 0.944 | 0.000 | 0.896 | 6 |
| tenant_big_2000 | 50 | 1993 | V1_wm16MB | hnsw:50 | 1.6 | 216.4 | 0.964 | 0.700 | 0.925 | 0 |
| tenant_big_2000 | 50 | 1993 | V1_wm64MB | hnsw:50 | 1.8 | 258.0 | 0.964 | 0.700 | 0.926 | 0 |
| tenant_big_2000 | 50 | 1993 | V1_wm256MB | hnsw:50 | 1.0 | 261.8 | 0.964 | 0.700 | 0.926 | 0 |
| tenant_big_2000 | 50 | 1993 | V2 | store_idx:50 | 1.4 | 2.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2000 | 50 | 1993 | V3 | hnsw_tenant:50 | 1.4 | 15.0 | 0.964 | 0.700 | 0.968 | 0 |
| tenant_big_8000 | 50 | 7994 | exact | store_idx:50 | 3.1 | 4.9 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_mst_default | hnsw:50 | 1.0 | 12.6 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_mst_100k | hnsw:50 | 0.7 | 3.4 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_mst_1m | hnsw:50 | 0.7 | 2.9 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1forced_mst_default | hnsw:50 | 0.8 | 8.5 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1forced_mst_100k | hnsw:50 | 0.8 | 9.2 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1forced_mst_1m | hnsw:50 | 0.7 | 3.1 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_wm16MB | hnsw:50 | 0.9 | 6.5 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_wm64MB | hnsw:50 | 1.1 | 5.5 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V1_wm256MB | hnsw:50 | 0.7 | 2.9 | 0.942 | 0.600 | 0.938 | 0 |
| tenant_big_8000 | 50 | 7994 | V2 | store_idx:50 | 3.9 | 8.9 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_8000 | 50 | 7994 | V3 | hnsw_tenant:50 | 1.3 | 33.6 | 0.958 | 0.000 | 0.957 | 2 |
| tenant_big_25000 | 50 | 24992 | exact | store_idx:50 | 9.2 | 14.3 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_mst_default | hnsw:50 | 2.3 | 27.4 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_mst_100k | hnsw:50 | 0.9 | 6.2 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_mst_1m | hnsw:50 | 0.9 | 4.0 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1forced_mst_default | hnsw:50 | 1.0 | 6.3 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1forced_mst_100k | hnsw:50 | 1.0 | 4.1 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1forced_mst_1m | hnsw:50 | 1.0 | 3.9 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_wm16MB | hnsw:50 | 1.1 | 5.3 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_wm64MB | hnsw:50 | 1.7 | 18.0 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V1_wm256MB | hnsw:50 | 0.9 | 3.0 | 0.930 | 0.100 | 0.916 | 0 |
| tenant_big_25000 | 50 | 24992 | V2 | store_idx:50 | 9.4 | 11.8 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_25000 | 50 | 24992 | V3 | hnsw_tenant:50 | 1.8 | 28.9 | 0.960 | 0.200 | 0.947 | 0 |
| tenant_big_60000 | 50 | 59993 | exact | store_idx:50 | 17.3 | 23.8 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_60000 | 50 | 59993 | V1_mst_default | hnsw:50 | 1.7 | 10.8 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1_mst_100k | hnsw:50 | 1.0 | 1.7 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1_mst_1m | hnsw:50 | 1.0 | 3.5 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1forced_mst_default | hnsw:50 | 1.1 | 4.5 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1forced_mst_100k | hnsw:50 | 1.1 | 2.9 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1forced_mst_1m | hnsw:50 | 1.0 | 3.5 | 0.930 | 0.000 | 0.906 | 1 |
| tenant_big_60000 | 50 | 59993 | V1_wm16MB | hnsw:50 | 1.5 | 2.8 | 0.950 | 0.600 | 0.924 | 0 |
| tenant_big_60000 | 50 | 59993 | V1_wm64MB | hnsw:50 | 2.0 | 6.6 | 0.950 | 0.600 | 0.924 | 0 |
| tenant_big_60000 | 50 | 59993 | V1_wm256MB | hnsw:50 | 1.4 | 2.3 | 0.950 | 0.600 | 0.924 | 0 |
| tenant_big_60000 | 50 | 59993 | V2 | store_idx:50 | 19.7 | 24.7 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_60000 | 50 | 59993 | V3 | hnsw_tenant:50 | 1.7 | 7.7 | 0.940 | 0.000 | 0.911 | 0 |

| policy (all sets above) | T (chunks) | p50 ms | p95 ms | recall@10 mean | recall@10 min | short |
|---|---|---|---|---|---|---|
| exact if < T else V1_mst_default | never | 3.2 | 168.4 | 0.898 | 0.000 | 93 |
| exact if < T else V1_mst_default | 1000 | 2.1 | 21.1 | 0.960 | 0.000 | 10 |
| exact if < T else V1_mst_default | 5000 | 2.0 | 13.2 | 0.965 | 0.000 | 3 |
| exact if < T else V1_mst_default | 10000 | 2.2 | 13.1 | 0.969 | 0.000 | 3 |
| exact if < T else V1_mst_default | 25000 | 2.4 | 13.1 | 0.975 | 0.000 | 3 |
| exact if < T else V1_mst_default | 50000 | 2.6 | 19.0 | 0.980 | 0.000 | 3 |
| exact if < T else V1_mst_default | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | never | 1.0 | 160.1 | 0.902 | 0.000 | 90 |
| exact if < T else V1_mst_100k | 1000 | 1.2 | 6.2 | 0.960 | 0.000 | 9 |
| exact if < T else V1_mst_100k | 5000 | 1.2 | 5.5 | 0.965 | 0.000 | 3 |
| exact if < T else V1_mst_100k | 10000 | 1.4 | 5.5 | 0.969 | 0.000 | 3 |
| exact if < T else V1_mst_100k | 25000 | 1.6 | 9.4 | 0.975 | 0.000 | 3 |
| exact if < T else V1_mst_100k | 50000 | 1.8 | 17.2 | 0.980 | 0.000 | 3 |
| exact if < T else V1_mst_100k | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | never | 1.0 | 95.3 | 0.902 | 0.000 | 90 |
| exact if < T else V1_mst_1m | 1000 | 1.2 | 6.5 | 0.960 | 0.000 | 9 |
| exact if < T else V1_mst_1m | 5000 | 1.2 | 5.2 | 0.965 | 0.000 | 3 |
| exact if < T else V1_mst_1m | 10000 | 1.3 | 5.2 | 0.969 | 0.000 | 3 |
| exact if < T else V1_mst_1m | 25000 | 1.5 | 9.4 | 0.975 | 0.000 | 3 |
| exact if < T else V1_mst_1m | 50000 | 1.7 | 17.2 | 0.980 | 0.000 | 3 |
| exact if < T else V1_mst_1m | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm16MB | never | 1.6 | 610.0 | 0.915 | 0.100 | 75 |
| exact if < T else V1_wm16MB | 1000 | 1.5 | 7.9 | 0.967 | 0.100 | 0 |
| exact if < T else V1_wm16MB | 5000 | 1.5 | 6.1 | 0.970 | 0.100 | 0 |
| exact if < T else V1_wm16MB | 10000 | 1.6 | 6.0 | 0.974 | 0.100 | 0 |
| exact if < T else V1_wm16MB | 25000 | 1.8 | 9.8 | 0.980 | 0.400 | 0 |
| exact if < T else V1_wm16MB | 50000 | 1.9 | 17.5 | 0.985 | 0.400 | 0 |
| exact if < T else V1_wm16MB | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm16MB | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm64MB | never | 1.8 | 5000.0 | 0.873 | 0.000 | 69 |
| exact if < T else V1_wm64MB | 1000 | 1.7 | 9.2 | 0.967 | 0.100 | 0 |
| exact if < T else V1_wm64MB | 5000 | 1.6 | 6.6 | 0.970 | 0.100 | 0 |
| exact if < T else V1_wm64MB | 10000 | 1.7 | 6.5 | 0.974 | 0.100 | 0 |
| exact if < T else V1_wm64MB | 25000 | 1.8 | 9.8 | 0.980 | 0.400 | 0 |
| exact if < T else V1_wm64MB | 50000 | 2.1 | 17.3 | 0.985 | 0.400 | 0 |
| exact if < T else V1_wm64MB | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm64MB | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm256MB | never | 1.0 | 5000.0 | 0.831 | 0.000 | 77 |
| exact if < T else V1_wm256MB | 1000 | 1.2 | 6.0 | 0.967 | 0.100 | 0 |
| exact if < T else V1_wm256MB | 5000 | 1.2 | 5.1 | 0.970 | 0.100 | 0 |
| exact if < T else V1_wm256MB | 10000 | 1.3 | 5.1 | 0.974 | 0.100 | 0 |
| exact if < T else V1_wm256MB | 25000 | 1.6 | 9.6 | 0.980 | 0.400 | 0 |
| exact if < T else V1_wm256MB | 50000 | 1.8 | 17.2 | 0.985 | 0.400 | 0 |
| exact if < T else V1_wm256MB | 100000 | 8.1 | 32.2 | 1.000 | 1.000 | 0 |
| exact if < T else V1_wm256MB | all exact | 8.1 | 32.2 | 1.000 | 1.000 | 0 |

| tenant-only leaf sets, chunks in set | n | exact p50 | exact p95 | V1 forced HNSW p50 (mst default) | V1 forced r10 | V1 forced mst_1m p50 | r10 | V3 tenant HNSW p50 | V3 r10 |
|---|---|---|---|---|---|---|---|---|---|
| 0-200 | 94 | 1.9 | 6.5 | 42.1 | 0.643 | 77.8 | 0.667 | 41.5 | 0.782 |
| 200-1000 | 6 | 1.8 | 3.2 | 31.1 | 0.933 | 22.8 | 0.933 | 8.1 | 1.000 |
| 1000-5000 | 50 | 1.4 | 6.2 | 1.7 | 0.944 | 1.7 | 0.944 | 1.4 | 0.964 |
| 5000-20000 | 50 | 3.1 | 4.9 | 0.8 | 0.942 | 0.7 | 0.942 | 1.3 | 0.958 |
| 20000-50000 | 50 | 9.2 | 14.3 | 1.0 | 0.930 | 1.0 | 0.930 | 1.8 | 0.960 |
| 50000-200000 | 50 | 17.3 | 23.8 | 1.1 | 0.930 | 1.0 | 0.930 | 1.7 | 0.940 |
