# Quantum-Machine-Learning

This repository contains the code and experiments for the paper "Bridging Classical and Quantum Machine Learning: Knowledge Transfer From Classical to Quantum Neural Networks Using Knowledge Distillation".

[![arXiv](https://img.shields.io/badge/arXiv-2311.13810-b31b1b.svg)](https://arxiv.org/abs/2311.13810)

## Table of Contents

1. [Motivation](#motivation)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Experiments](#experiments)
   - [MNIST](#mnist)
   - [FashionMNIST](#fashionmnist)
5. [Results](#results)
6. [Contributing](#contributing)
7. [Citation](#citation)
8. [License](#license)

## Motivation

Quantum neural networks have shown promise in surpassing classical neural networks in tasks like image classification when using a similar number of learnable parameters. However, the development and optimization of quantum models face challenges such as qubit instability and limited availability. This project introduces a novel method to transfer knowledge from classical to quantum neural networks using knowledge distillation, effectively bridging the gap between classical machine learning and emerging quantum computing techniques.

Our approach adapts classical convolutional neural network (CNN) architectures like LeNet and AlexNet to serve as teacher networks, facilitating the training of student quantum models. This method yields significant performance improvements for quantum models by solely depending on classical CNNs, eliminating the need for cumbersome training of large quantum models in resource-constrained settings.

## Project Structure

```
.
├── FashionMNIST Experiments
│   ├── Baseline students
│   ├── Distillation on students
│   └── Teachers
├── MNIST Experiments
│   ├── Baseline students
│   ├── Distillation on students
│   └── Teachers
└── README.md
```

Each subdirectory contains Jupyter notebooks for the respective experiments.

## Setup

To run the experiments, you'll need to set up your environment with the necessary dependencies. Follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/quantum-classical-knowledge-distillation.git
   cd quantum-classical-knowledge-distillation
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: The `requirements.txt` file should include all necessary libraries, such as PyTorch, TensorFlow Quantum, and other dependencies.

4. Install Jupyter Notebook or JupyterLab:
   ```
   pip install jupyter
   ```

## Experiments

### MNIST

The MNIST experiments are located in the `MNIST Experiments` directory. To reproduce the results:

1. Navigate to the `MNIST Experiments` directory.
2. Run the notebooks in the following order:
   - Teachers: Train the classical CNN models.
   - Baseline students: Train the quantum models without distillation.
   - Distillation on students: Apply knowledge distillation from classical to quantum models.

### FashionMNIST

The FashionMNIST experiments follow a similar structure in the `FashionMNIST Experiments` directory. Repeat the steps as described for MNIST.

## Results

Our approach yields significant performance improvements for quantum models:

- MNIST dataset: Average accuracy improvement of 0.80%
- FashionMNIST dataset: Average accuracy improvement of 5.40%

For detailed results and analysis, please refer to the full paper.

## Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests.

## Citation

If you use this code or our results in your research, please cite our paper:

```bibtex
@article{hasan2023bridging,
  title={Bridging Classical and Quantum Machine Learning: Knowledge Transfer From Classical to Quantum Neural Networks Using Knowledge Distillation},
  author={Hasan, Mohammad Junayed and Mahdy, MRC},
  journal={arXiv preprint arXiv:2311.13810},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Copyright © 2023 [Your Name/Organization]. All rights reserved.
