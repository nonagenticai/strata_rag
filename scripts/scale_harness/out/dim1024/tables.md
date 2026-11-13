chunks=197262 dim=1024 tenants=300+big(1000,2500,5000,10000,20000) load=15s global_hnsw_build=78s hnsw=1.62 GB db=2.86 GB

| set | n | chunks in set (median) | path | plan | p50 ms | p95 ms | recall@10 mean | recall@10 min | recall@50 mean | short (rows < min(50, set)) |
|---|---|---|---|---|---|---|---|---|---|---|
| tenant_scope | 100 | 37 | exact | store_idx:100 | 1.9 | 4.6 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 37 | V1_mst_default | store_idx:100 | 0.6 | 2.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 37 | V1_mst_100k | store_idx:100 | 0.7 | 1.3 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 37 | V1_mst_1m | store_idx:100 | 0.8 | 2.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_scope | 100 | 37 | V1forced_mst_default | hnsw:100 | 97.8 | 216.7 | 0.674 | 0.100 | 0.386 | 82 |
| tenant_scope | 100 | 37 | V1forced_mst_100k | hnsw:100 | 518.8 | 860.7 | 0.934 | 0.200 | 0.825 | 45 |
| tenant_scope | 100 | 37 | V1forced_mst_1m | hnsw:100 | 694.7 | 1265.7 | 0.956 | 0.250 | 0.884 | 35 |
| platform_job | 100 | 43 | exact | store_idx:100 | 2.0 | 5.5 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1_mst_default | store_idx:100 | 0.6 | 1.1 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1_mst_100k | store_idx:100 | 0.6 | 4.4 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1_mst_1m | store_idx:100 | 0.6 | 2.2 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1forced_mst_default | hnsw:100 | 90.6 | 155.2 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1forced_mst_100k | hnsw:100 | 511.5 | 840.3 | 1.000 | 1.000 | 1.000 | 0 |
| platform_job | 100 | 43 | V1forced_mst_1m | hnsw:100 | 625.6 | 1030.8 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | exact | store_idx:100 | 21.2 | 65.7 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1_mst_default | store_idx:100 | 16.7 | 33.0 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1_mst_100k | store_idx:100 | 16.2 | 29.0 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1_mst_1m | store_idx:100 | 16.3 | 31.4 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1forced_mst_default | hnsw:100 | 5.6 | 43.4 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1forced_mst_100k | hnsw:100 | 1.3 | 4.4 | 1.000 | 1.000 | 1.000 | 0 |
| platform_scope | 100 | 7521 | V1forced_mst_1m | hnsw:100 | 1.3 | 4.3 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 7365 | exact | store_idx:100 | 18.6 | 41.2 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 7365 | V1_mst_default | store_idx:100 | 16.0 | 23.4 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 7365 | V1_mst_100k | store_idx:100 | 16.1 | 22.4 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 7365 | V1_mst_1m | store_idx:100 | 16.2 | 20.9 | 1.000 | 1.000 | 1.000 | 0 |
| mixed | 100 | 7365 | V1forced_mst_default | hnsw:100 | 6.4 | 23.9 | 0.999 | 0.900 | 0.995 | 0 |
| mixed | 100 | 7365 | V1forced_mst_100k | hnsw:100 | 1.7 | 4.6 | 0.999 | 0.900 | 0.995 | 0 |
| mixed | 100 | 7365 | V1forced_mst_1m | hnsw:100 | 1.7 | 5.4 | 0.999 | 0.900 | 0.995 | 0 |
| tenant_big_1000 | 50 | 996 | exact | store_idx:50 | 2.7 | 7.1 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_1000 | 50 | 996 | V1_mst_default | store_idx:50 | 2.3 | 3.3 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_1000 | 50 | 996 | V1_mst_100k | store_idx:50 | 2.7 | 4.3 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_1000 | 50 | 996 | V1_mst_1m | store_idx:50 | 2.7 | 4.6 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_1000 | 50 | 996 | V1forced_mst_default | hnsw:50 | 6.6 | 119.4 | 1.000 | 1.000 | 0.875 | 21 |
| tenant_big_1000 | 50 | 996 | V1forced_mst_100k | hnsw:50 | 3.4 | 509.5 | 1.000 | 1.000 | 0.950 | 0 |
| tenant_big_1000 | 50 | 996 | V1forced_mst_1m | hnsw:50 | 3.2 | 385.1 | 1.000 | 1.000 | 0.954 | 0 |
| tenant_big_2500 | 50 | 2494 | exact | store_idx:50 | 6.6 | 9.8 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1_mst_default | store_idx:50 | 6.6 | 12.3 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1_mst_100k | store_idx:50 | 6.5 | 14.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1_mst_1m | store_idx:50 | 6.5 | 12.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1forced_mst_default | hnsw:50 | 1.7 | 10.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1forced_mst_100k | hnsw:50 | 1.2 | 4.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_2500 | 50 | 2494 | V1forced_mst_1m | hnsw:50 | 1.2 | 2.8 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_5000 | 50 | 4993 | exact | store_idx:50 | 12.4 | 34.5 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_5000 | 50 | 4993 | V1_mst_default | store_idx:50 | 12.1 | 24.5 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_5000 | 50 | 4993 | V1_mst_100k | store_idx:50 | 12.3 | 16.0 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_5000 | 50 | 4993 | V1_mst_1m | store_idx:50 | 12.2 | 17.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_5000 | 50 | 4993 | V1forced_mst_default | hnsw:50 | 1.9 | 12.8 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_5000 | 50 | 4993 | V1forced_mst_100k | hnsw:50 | 1.4 | 3.8 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_5000 | 50 | 4993 | V1forced_mst_1m | hnsw:50 | 1.3 | 4.5 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_10000 | 50 | 9992 | exact | store_idx:50 | 21.9 | 34.6 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_10000 | 50 | 9992 | V1_mst_default | store_idx:50 | 22.3 | 27.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_10000 | 50 | 9992 | V1_mst_100k | store_idx:50 | 21.7 | 29.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_10000 | 50 | 9992 | V1_mst_1m | store_idx:50 | 22.6 | 29.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_10000 | 50 | 9992 | V1forced_mst_default | hnsw:50 | 1.7 | 16.1 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_10000 | 50 | 9992 | V1forced_mst_100k | hnsw:50 | 1.4 | 2.7 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_10000 | 50 | 9992 | V1forced_mst_1m | hnsw:50 | 1.5 | 2.4 | 0.996 | 0.900 | 0.996 | 0 |
| tenant_big_20000 | 50 | 19995 | exact | store_idx:50 | 41.2 | 63.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_20000 | 50 | 19995 | V1_mst_default | store_idx:50 | 42.2 | 54.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_20000 | 50 | 19995 | V1_mst_100k | store_idx:50 | 42.5 | 53.4 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_20000 | 50 | 19995 | V1_mst_1m | store_idx:50 | 41.9 | 48.2 | 1.000 | 1.000 | 1.000 | 0 |
| tenant_big_20000 | 50 | 19995 | V1forced_mst_default | hnsw:50 | 3.0 | 13.1 | 0.992 | 0.800 | 0.989 | 0 |
| tenant_big_20000 | 50 | 19995 | V1forced_mst_100k | hnsw:50 | 1.9 | 4.1 | 0.992 | 0.800 | 0.989 | 0 |
| tenant_big_20000 | 50 | 19995 | V1forced_mst_1m | hnsw:50 | 1.9 | 2.4 | 0.992 | 0.800 | 0.989 | 0 |

| policy (all sets above) | T (chunks) | p50 ms | p95 ms | recall@10 mean | recall@10 min | short |
|---|---|---|---|---|---|---|
| exact if < T else V1_mst_default | never | 12.3 | 41.9 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 1000 | 12.4 | 41.9 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 5000 | 12.5 | 42.0 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 10000 | 12.5 | 43.1 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 25000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 50000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | 100000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_default | all exact | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | never | 12.4 | 41.7 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 1000 | 12.4 | 41.7 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 5000 | 12.5 | 41.5 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 10000 | 12.5 | 43.5 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 25000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 50000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | 100000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_100k | all exact | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | never | 12.5 | 41.5 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 1000 | 12.8 | 41.5 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 5000 | 12.5 | 41.6 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 10000 | 12.5 | 43.6 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 25000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 50000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | 100000 | 12.5 | 42.8 | 1.000 | 1.000 | 0 |
| exact if < T else V1_mst_1m | all exact | 12.5 | 42.8 | 1.000 | 1.000 | 0 |

| tenant-only leaf sets, chunks in set | n | exact p50 | exact p95 | V1 forced HNSW p50 (mst default) | V1 forced r10 | V1 forced mst_1m p50 | r10 | V3 tenant HNSW p50 | V3 r10 |
|---|---|---|---|---|---|---|---|---|---|
| 0-200 | 99 | 1.8 | 4.6 | 97.8 | 0.671 | 694.7 | 0.955 | - | - |
| 200-1000 | 51 | 2.7 | 7.1 | 6.6 | 1.000 | 3.2 | 1.000 | - | - |
| 1000-5000 | 100 | 10.6 | 26.4 | 1.9 | 0.998 | 1.3 | 0.998 | - | - |
| 5000-20000 | 100 | 38.1 | 57.8 | 2.2 | 0.994 | 1.6 | 0.994 | - | - |
