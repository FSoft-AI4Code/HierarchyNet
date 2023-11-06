### HierarchyNet
***
All source code are written in Python. Besides Pytorch, we also use many other libraries such as DGL, scikit-learn, pandas, jsonlines
1. Datasets: all the datasets used in the paper are publicly accessible.
2. Data preprocessing: folder *preprocessing* is used to prepare data in the proper format before training. Go to this folder for more information.
3. Modify the configuration file in the folder c2nl/configs such that all the paths are valid
4. Train model
```
cd c2nl
bash main/train.sh
```
