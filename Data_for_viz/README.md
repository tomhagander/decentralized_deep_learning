# Data for visualisation

This directory contains outputs from `toy_regression_visualization.ipynb`.

## Replay data

`toy_regression_three_strategies.json` contains three comparable strategies. It is the browser-ready conversion of the saved Python client pickles:

- `dac`: adaptive communication using inverse-training-loss similarity and prior-weighted aggregation.
- `oracle`: communication only with clients in the same known cluster.
- `random`: communication with any other client.

Each strategy contains `states` for round 0 through the final communication round. A state contains:

- `clusterAverageValidationLoss`: mean validation MSE for each cluster.
- `clusterAverageGuess`: mean prediction for the fixed probe input `[1, 0, ..., 0]`.
- `nodes`: client IDs, cluster IDs, and validation losses.
- `communications`: directed communication events for that round.

DAC states also contain `similarityMatrix`, a 30 by 30 client-to-client matrix. Row `i`, column `j` is the similarity score known by client `i` for client `j` at that point in the experiment. Oracle and random states intentionally omit this field because they do not calculate DAC similarities.

## Generated diagnostics

The notebook also writes comparison and DAC-only diagnostic plots, including:

- `strategy_comparison.png`
- `dac_sample_client_training_validation_loss.png`
- `dac_final_similarity_heatmap.png`
- `dac_final_neighbor_prior_heatmap.png`
- `dac_communication_count_heatmap.png`

The notebook is configured for 30 clients, 3 clusters, 30 communication rounds, and 3 sampled neighbors per client. The currently committed converted replay was generated from saved pickle files containing 50 rounds; rerunning the notebook will generate a fresh 30-round replay.

The website should load the JSON file only. The `.pkl` files are Python analysis artifacts and are not required in the browser. `toy_regression_three_strategies_from_pickle.json` is an identical conversion kept as an explicitly named backup.
