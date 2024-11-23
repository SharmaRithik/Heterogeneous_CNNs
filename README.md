# Heterogeneous_CNNs

Dense: g++ -std=c++17 -O3 cifar_dense.cpp

sparse: g++ -std=c++17 -O3 cifar_sparse.cpp

Test: ./test_cifar10.sh (use g++ -std=c++17 -O3 <filename>, to get a.out)

### Model Training and Pruning Log

```bash
riksharm@shrike:~/Heterogeneous_CNNs/train$ python3 train_and_extract.py 
Files already downloaded and verified
Files already downloaded and verified
Using device: cuda
Pruning model with 99.0% sparsity...
Starting training...
Epoch [1/5], Step [100/391], Loss: 1.4162
Epoch [1/5], Step [200/391], Loss: 1.5587
Epoch [1/5], Step [300/391], Loss: 1.1053
Epoch [1/5], Loss: 1.3852, Accuracy: 49.70%
Epoch [2/5], Step [100/391], Loss: 0.8486
Epoch [2/5], Step [200/391], Loss: 0.8484
Epoch [2/5], Step [300/391], Loss: 0.8832
Epoch [2/5], Loss: 0.9472, Accuracy: 66.50%
Epoch [3/5], Step [100/391], Loss: 0.6119
Epoch [3/5], Step [200/391], Loss: 0.8944
Epoch [3/5], Step [300/391], Loss: 0.7893
Epoch [3/5], Loss: 0.7812, Accuracy: 72.64%
Epoch [4/5], Step [100/391], Loss: 0.7683
Epoch [4/5], Step [200/391], Loss: 0.8267
Epoch [4/5], Step [300/391], Loss: 0.5678
Epoch [4/5], Loss: 0.6737, Accuracy: 76.54%
Epoch [5/5], Step [100/391], Loss: 0.5138
Epoch [5/5], Step [200/391], Loss: 0.6076
Epoch [5/5], Step [300/391], Loss: 0.6157
Epoch [5/5], Loss: 0.5900, Accuracy: 79.26%
Test Accuracy: 74.97%
```

### Dense Data Sparsity

```bash
features_0_weight.txt: Sparsity = 0.00%
features_3_weight.txt: Sparsity = 0.00%
features_6_weight.txt: Sparsity = 0.00%
features_8_weight.txt: Sparsity = 0.00%
features_10_weight.txt: Sparsity = 0.00%
features_0_bias.txt: Sparsity = 0.00%
features_3_bias.txt: Sparsity = 0.00%
features_6_bias.txt: Sparsity = 0.00%
features_8_bias.txt: Sparsity = 0.00%
features_10_bias.txt: Sparsity = 0.00%
classifier_bias.txt: Sparsity = 0.00%
classifier_weight.txt: Sparsity = 0.00%
```

### Sparse Data Sparsity

```bash
features_0_weight.txt: Sparsity = 73%
features_3_weight.txt: Sparsity = 88%
features_6_weight.txt: Sparsity = 83%
features_8_weight.txt: Sparsity = 84%
features_10_weight.txt: Sparsity = 79%
features_0_bias.txt: Sparsity = 0.00%
features_3_bias.txt: Sparsity = 0.00%
features_6_bias.txt: Sparsity = 0.00%
features_8_bias.txt: Sparsity = 0.00%
features_10_bias.txt: Sparsity = 0.00%
classifier_bias.txt: Sparsity = 0.00%
classifier_weight.txt: Sparsity = 50%
```
