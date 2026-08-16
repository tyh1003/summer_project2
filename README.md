

# SimCLR Self-Supervised Learning on CIFAR-10

This project explores **self-supervised representation learning using SimCLR** on CIFAR-10. A modified ResNet-18 is trained without class labels using contrastive learning, and the learned representations are evaluated through kNN monitoring and linear probing.

The experiments also compare SimCLR with supervised learning and a random backbone, investigate the effects of temperature and the projection head, and evaluate transferability to CIFAR-100.

## Experimental Setup

The experiments were conducted on an **NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB VRAM and CUDA 13.2**. 

For the SimCLR baseline:

| Setting         | Value                  |
| --------------- | ---------------------- |
| Dataset         | CIFAR-10               |
| Backbone        | Modified ResNet-18     |
| Pretrained      | No                     |
| Backbone Output | 512                    |
| Projector       | 512 → 512 → 128        |
| Loss            | NT-Xent Loss           |
| Optimizer       | Adam                   |
| Learning Rate   | 0.0003                 |
| Weight Decay    | 0.000001               |
| Temperature     | 0.5                    |
| Batch Size      | 256                    |
| SSL Epochs      | 200                    |
| kNN Monitor     | k = 20, every 5 epochs |

For linear probing, the backbone is frozen and only a `512 → 10` linear classifier is trained for 100 epochs using Adam with a learning rate of `0.001`. 

---

## 1. SimCLR Baseline

The SimCLR model was trained from scratch on CIFAR-10 for **200 epochs** using NT-Xent loss.

### Results

| Metric                        |        Result |
| ----------------------------- | ------------: |
| Initial Loss                  |        5.2717 |
| Final Loss                    |    **4.4999** |
| Final kNN Accuracy            |    **84.96%** |
| Final Linear Probing Accuracy |    **86.77%** |
| Best Linear Probing Accuracy  |    **87.07%** |
| SSL Training Time             |   **4.20 hr** |
| Linear Probing Time           | **21.96 min** |

During training, the NT-Xent loss decreased from **5.2717 to 4.4999**, while kNN accuracy gradually increased to **84.96%**. The final linear probing accuracy reached **86.77%**, with a best accuracy of **87.07%**. 

---

## 2. Supervised Learning

A modified ResNet-18 was trained directly on CIFAR-10 labels using supervised learning. The model was trained from scratch for 200 epochs using cross-entropy loss. 

### Comparison

| Method                      | Test Accuracy | Training Time |
| --------------------------- | ------------: | ------------: |
| SimCLR SSL + Linear Probing |    **86.77%** |        4.5 hr |
| Supervised Learning         |    **92.43%** |        1.6 hr |

Supervised learning achieved **92.43%**, which is **5.66 percentage points higher** than SimCLR.

However, supervised learning directly uses class labels and updates the entire backbone. SimCLR learns representations without class labels, and a frozen SimCLR backbone still achieves **86.77%** using only a linear classifier. 

---

## 3. Random Backbone

To determine whether SimCLR actually learns useful representations, a randomly initialized modified ResNet-18 was completely frozen and evaluated using the same linear probing approach.

The linear classifier was trained for 100 epochs using Adam with a learning rate of `1e-3` and batch size 256. 

### Comparison

| Method              | Training                      | Test Accuracy | Training Time |
| ------------------- | ----------------------------- | ------------: | ------------: |
| Random Backbone     | Random initialization, frozen |    **41.83%** |        23 min |
| SimCLR SSL          | Self-supervised               |    **86.77%** |        4.5 hr |
| Supervised Learning | Supervised                    |    **92.43%** |        1.6 hr |

SimCLR improves accuracy from **41.83% to 86.77%**, a difference of **44.94 percentage points** compared with the random backbone.

This indicates that the performance does not come only from the ResNet-18 architecture. SimCLR actually learns discriminative image representations through self-supervised training. 

---

## 4. Temperature Ablation

Three temperature values were tested to investigate the effect of temperature in the NT-Xent loss.

|        Temperature | Final Loss | Final kNN Accuracy | Training Time |
| -----------------: | ---------: | -----------------: | ------------: |
|                0.1 | **0.4328** |             81.57% |       4.19 hr |
| **0.5 (Baseline)** |     4.4999 |         **84.96%** |       4.20 hr |
|                  5 |     6.0536 |             77.94% |       4.58 hr |

Although `T = 0.1` produced the lowest final loss (**0.4328**), it did not achieve the best representation performance. The baseline `T = 0.5` achieved the highest kNN accuracy at **84.96%**.

Therefore, NT-Xent loss values obtained under different temperatures cannot be directly used to compare representation quality. 

---

## 5. Projector Output

The baseline uses the **512-dimensional backbone output** for downstream linear probing. This experiment instead uses the **128-dimensional projector output** as the representation.

| Representation           |               Test Accuracy |
| ------------------------ | --------------------------: |
| Backbone Output (512-d)  |                  **86.77%** |
| Projector Output (128-d) |                  **82.92%** |
| Difference               | **−3.85 percentage points** |

Using the projector output reduces linear probing accuracy from **86.77% to 82.92%**.

This suggests that the backbone output retains features that are more suitable for downstream classification than the projector output. 

---

## 6. No Projector

To evaluate the role of the projection head during contrastive training, SimCLR was also trained without the projector.

| Metric                  | Baseline: With Projector | No Projector | Difference |
| ----------------------- | -----------------------: | -----------: | ---------: |
| Final Loss              |               **4.4999** |       4.5956 |    +0.0957 |
| Final kNN Accuracy      |               **84.96%** |       82.64% |   −2.32 pp |
| Linear Probing Accuracy |               **86.77%** |       83.64% |   −3.13 pp |
| SSL Training Time       |                  4.20 hr |      4.16 hr |    Similar |

Removing the projector decreased kNN accuracy from **84.96% to 82.64%** and linear probing accuracy from **86.77% to 83.64%**.

Therefore, the projector helps SimCLR learn better backbone representations during contrastive training, even though the projector output itself is not the best representation for downstream classification. 

---

## 7. CIFAR-100 Transfer

The learned CIFAR-10 representations were transferred to **CIFAR-100** to evaluate whether they remain useful on a new dataset.

| Representation      | CIFAR-100 Accuracy |
| ------------------- | -----------------: |
| Random Backbone     |         **18.57%** |
| SimCLR SSL          |         **50.05%** |
| Supervised Learning |         **49.24%** |

On CIFAR-10, supervised learning performs better than SimCLR:

**92.43% > 86.77%**

However, after transferring to CIFAR-100:

**SimCLR 50.05% ≈ Supervised 49.24%**

and both are substantially better than the random backbone at **18.57%**. 

This result suggests that the representation learned by SimCLR can transfer effectively to a new dataset and shows good generalization ability. 

---

## Discussion

The experiments provide four main observations.

**1. SimCLR learns meaningful representations.**
The accuracy increases from **41.83% with a random backbone to 86.77% with SimCLR**, while supervised learning reaches **92.43%**. This shows that self-supervised contrastive training learns useful visual features even without class labels. 

**2. Lower contrastive loss does not necessarily mean better representations.**
At `T = 0.1`, the final loss is only **0.4328**, but kNN accuracy is **81.57%**. In contrast, `T = 0.5` has a higher loss of **4.4999** but achieves the best kNN accuracy of **84.96%**. 

**3. The projector is useful during training, but its output is not the best downstream representation.**
Training **with a projector performs better than without a projector**, while during evaluation the **backbone output performs better than the projector output**. 

**4. SimCLR representations transfer well to a new dataset.**
On CIFAR-100, the random backbone achieves **18.57%**, SimCLR achieves **50.05%**, and supervised learning achieves **49.24%**. This suggests that SimCLR learns representations with good transferability beyond the original CIFAR-10 dataset. 

---

## Conclusion

SimCLR successfully learns useful visual representations from CIFAR-10 without using class labels. Although supervised learning achieves higher accuracy on the original CIFAR-10 task, SimCLR substantially outperforms a random backbone and shows comparable performance to supervised representations when transferred to CIFAR-100.

The experiments also show that **temperature selection affects representation quality**, and that the **projection head is important during contrastive training while the backbone output is more suitable for downstream classification**.
