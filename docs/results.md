
## Results
Joint prediction results on the [Waymo Open Motion](https://arxiv.org/abs/2104.10133) dataset (Validation, Interactive Split, Vehicles-Only). Please refer to the [paper](https://openreview.net/pdf?id=Eyb4e3GBuuR) for more details.

| **Method**   | **minADE ↑**    | **minFDE ↑**    | **minSADE ↑**   | **minSFDE ↓**   | **OR ↓**          |
| ------------ | --------------- | --------------- | --------------- | --------------- | ----------------- |
| V-LSTM + SC  | 3.08 ± 0.04     | 6.75 ± 0.09     | 3.62 ± 0.04     | 8.50 ± 0.08     | 0.047 ± 0.002     |
| V-LSTM + EPO | **2.62 ± 0.01** | **5.83 ± 0.06** | **3.10 ± 0.01** | **7.35 ± 0.04** | **0.045 ± 0.004** |

The code has been restructured for the upload. We reproduced these with the current codebase.

Supported by the Federal Ministry for Economic Affairs and Climate Action on the basis of a decision by the German Bundestag and the European Union in the Project KISSaF - AI-based Situation Interpretation for Automated Driving.