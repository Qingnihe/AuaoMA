# AutoAno

## Dataset Preparation

1. Navigate to the `DataSets` folder and refer to its `README.md` for instructions on downloading the raw dataset.

2. Run the following scripts to prepare the data:

   ```bash
   cd DataSets
   python get_data_asd/smd.py
   python create_data_asd/smd.py
   python create_anomaly_asd/smd.py
   ```

## Launch the Experiment with NNI

1. Navigate to the `NAS_HPO` directory:

   ```bash
   cd ../NAS_HPO
   ```

2. Launch the experiment using NNI:

   ```bash
   nnictl create --config test.yml
   ```
