

![logo](https://github.com/vasanthgx/ablation_study/blob/main/images/logo.gif)


# Project Title


**Ablation Study - Effect of depth and breadth of Neural Networks on performance**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>






## Introduction

The fashion industry has seen a significant transformation with the integration of technology, and one such intersection is in the realm of image classification. The Fashion MNIST (FMNIST) dataset serves as a benchmark for evaluating machine learning and deep learning algorithms on complex visual data. This project explores the effectiveness of various neural network architectures using an ablation study to determine the optimal configuration for image classification tasks.


## Project Overview

This project aims to perform an ablation study on different Sequential Neural Network configurations to identify the best-performing model for the Fashion MNIST dataset. The study involves experimenting with varying numbers of hidden layers, units per layer, and dropout layers to observe their impact on model performance. Six different models are trained and evaluated based on their accuracy, total parameters, and overall efficiency.

## Key Features

- **Ablation Study**: Systematic evaluation of different neural network architectures.
- **Performance Metrics**: Comparison based on accuracy, number of parameters, and computational complexity.
- **Model Variants**: Includes models with different depths and complexities, ranging from simple to highly complex architectures.
- **Dropout Layers**: Inclusion of dropout layers in certain models to assess their impact on preventing overfitting.
- **Visualization**: Graphical representation of loss curves to understand the training dynamics of each model.
- **Tensor Board** : TensorBoard is employed as a crucial feature in this project to facilitate the comprehensive visualization and analysis of model performance. By integrating TensorBoard, we can effectively monitor the training process in real-time. Key logs captured include: Scalars, Images, Distributions , Graphs and Histograms.

### Implementation Details

The implementation involves the following steps:

1. **Data Preparation:**

Load the Fashion MNIST dataset.
Preprocess the data by normalizing the pixel values to a range of 0 to 1.

2. **Model Design:**

- Design six different Sequential Neural Network models with varying configurations:
	- Model 1: 3 hidden layers + dropout, 32 + 16 + 8 + 8 units.
	- Model 2: 3 hidden layers + dropout, 64 + 32 + 16 + 16 units.
	- Model 3: 1 hidden layer, 128 units.
	- Model 4: 3 hidden layers + dropout, 128 + 64 + 32 + 32 units.
	- Model 5: 2 hidden layers, 256 + 128 units.
	- Model 6: 5 hidden layers + dropout, 256 + 128 + 64 + 32 + 16 + 16 units.
	
3. **Training:**

- Train each model for 5 epochs.
- Use a consistent batch size and optimizer (e.g., Adam) across all models.
- Monitor the loss and accuracy during training.

4. **Evaluation:**

- Evaluate each model on the test set.
- Record accuracy, total parameters, and other relevant metrics.

5. Visualization:

- Plot loss curves for each model to visualize training and validation loss over epochs.
- Compare model performance based on the evaluation metrics.

### Dataset Details

[The original source of the dataset](http://yann.lecun.com/exdb/mnist/)

The Fashion MNIST dataset is a collection of 70,000 grayscale images of 28x28 pixels, distributed into 10 categories representing different types of clothing and accessories. It is intended to serve as a more challenging alternative to the traditional MNIST dataset of handwritten digits.

- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Classes:**
	1. T-shirt/top
	2. Trouser
	3. Pullover
	4. Dress
	5. Coat
	6. Sandal
	7. Shirt
	8. Sneaker
	9. Bag
	10. Ankle boot
Each image is associated with a single label, and the goal is to classify the images into their respective categories. The dataset is pre-divided into a training set and a test set, facilitating straightforward evaluation of model performance.

By following this structured approach, the project effectively explores the impact of different neural network configurations on the FMNIST dataset, providing insights into the optimal design choices for image classification tasks.

### FMNIST Dataset

 ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/fmnist.png)

### Analysis and Conclusion:

 ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/ablation-1.png)
 
 
 **Predictions of the labels from the 2nd Model**
 
 ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/predictions-1.png)
 
 ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/ablation-2.png)
 
 **Tensor Board Integration for visualizing and logging different experiments**
 
  ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/TB-2.png)
 
 
 ![alt text](https://github.com/vasanthgx/ablation_study/blob/main/images/ablation-3.png)

1. **Model Complexity and Performance:**

- The simplest model (Model 1) with fewer units and parameters performed the worst with 83.63% accuracy.

- Model 5, with moderate complexity (2 hidden layers with 256 and 128 units) and the highest accuracy (87.55%), indicates that increasing the number of units can significantly improve performance up to a certain point.

- Models with intermediate complexity (Models 2, 3, and 4) show that adding more hidden layers can improve accuracy, but the benefits diminish with too many layers and units (as seen in Model 4).
	
2. **Number of Parameters:**

- Models with higher parameters (Model 5 and Model 6) tend to perform better than simpler models. However, the performance gain diminishes as the model complexity increases beyond a certain threshold (Model 6 compared to Model 5).

3. **Dropout Layers:**

- Dropout layers are used in Models 1, 2, 4, and 6 to prevent overfitting. Models with dropout layers (2, 4, and 6) generally perform well, but the most complex model (Model 6) did not surpass Model 5 despite having dropout layers, suggesting a limit to the benefit of adding complexity.

4. **Optimal Model:**

- Model 5 appears to be the optimal configuration, balancing complexity and performance. It has the highest accuracy of 87.55% with a manageable number of parameters (235,146) and two hidden layers.

### Conclusion:

The study demonstrates that increasing the number of hidden layers and units generally improves the performance of neural networks on the FMNIST dataset up to a point. However, beyond a certain complexity, the gains in accuracy diminish. Dropout layers help in preventing overfitting and maintaining model performance. The optimal model found in this study (Model 5) suggests that a moderately complex architecture with sufficient units per layer provides the best performance. This finding highlights the importance of balancing model complexity and computational resources to achieve the best results.


## FAQ

### 1) Tensor Board Integration

TensorBoard is employed as a crucial feature in this project to facilitate the comprehensive visualization and analysis of model performance. By integrating TensorBoard, we can effectively monitor the training process in real-time. Key logs captured include:

- **Scalars:** Track metrics such as loss and accuracy over epochs, providing insight into the model's learning progress.
- **Images:** Visualize input images, allowing for inspection of the data being fed into the network.
- **Histograms:** Analyze the distribution of weights and biases, offering a deeper understanding of how model parameters evolve during training.
- **Distributions:** Examine the detailed statistical distribution of tensors, aiding in the identification of potential issues like vanishing or exploding gradients.
TensorBoard's interactive dashboards enhance the interpretability of complex training dynamics, making it an invaluable tool for optimizing neural network performance.




## Acknowledgements


 - [Tensorflow datasets - Fashion MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
 - [Image Segmentaion](https://www.tensorflow.org/tutorials/generative/autoencoder)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

