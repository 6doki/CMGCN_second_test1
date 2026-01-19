# CMGCN: Completion Multi-Graph Convolutional Network for incompletion Traffic Forecasting
This is the pytorch re-implement of the CMGCN model.
This folder concludes the code and data of our CMGCN model

## Structure:

* Data: including PeMS04 and PeMS08 with missing rate of 10% and 30%.

* raw_data: including PeMS04 and PeMS08 dataset after completing.

* evaluator: contains self-defined modules for our work, such as evaluate metrics

* executor: data pre-process

* utils: normalization

* model: implementation of our CMGCN model

* gmf: implementation of our CLN+ module


## Quick start

Put your data in <u>**CMGCN/raw_data**.</u> 

For example, if you want to run model on dataset PEMS04 with missing rate of 10%, put the file **<u>distance.csv</u>** and file <u>**PEMS04_10.npz**</u> in **<u>CMGCN/raw_data/PEMS04_10/</u>** .

Set appropriate value of parameter in **<u>CMGCN/config/*.json</u>**.

Run <u>**python gmf/main.py**</u> to run the CLN+ module completing PeMS datasets.

Then put the complete datasets,such as **<u>PEMS04_30.npz</u>** into corresponding folder, such as  **<u>raw_data/PEMS04_10/PEMS04_10.npz</u>**.

Run <u>**python main.py**</u> to run the model. 


