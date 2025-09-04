# PMIANLL: Defending against Poisoning Membership Inference Attacks with Noisy Label Learning

This is the official PyTorch implementation for PMIANLL Defending against Poisoning Membership Inference Attacks with Noisy Label Learning.

## Features

- Support for multiple datasets (CIFAR-10, CIFAR-100, STL-10, Tiny-ImageNet)
- Various model architectures (ResNet, VGG, etc.)

## Dependencies

| Requirement | Version |
|------------|---------|
| Python     | 3.6+    |
| PyTorch    | Latest  |
| torchvision| Latest  |
| matplotlib | Latest  |
| scikit-learn| Latest |
| tqdm       | Latest  |

## Environment Setup

You can set up the environment using conda:
```bash
conda env create -f environment.yml
conda activate pmianll
```

## Datasets

Currently supported datasets:
- CIFAR-10
- CIFAR-100
- STL-10
- Tiny-ImageNet

Datasets will be automatically downloaded when running the training scripts.

## Running Instructions

### CIFAR-10 Experiments

#### 1. Train Clean Model
```bash
python3 train.py --clean_model True --experiment-name 'MD-DYR-SH' --epochs 300 \
    --model_name "resnet18" --dataset "CIFAR10" --device "cuda:0" --exp 0
```

#### 2. Train Poisoned Model (No Defense)
```bash
python3 train.py --Mixup 'None' --experiment-name 'MD-DYR-SH' --epochs 300 \
    --M 100 250 --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 2.0 --device "cuda:0" --exp 1
```

Deploy Membership Inference Attack:
```bash
python3 privacy_risks_evaluations/metrics_based_MIA.py --Mixup 'None' \
    --experiment-name 'MD-DYR-SH' --epochs 300 --M 100 250 --distribution gaussian \
    --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 2.0 --device "cuda:0" --exp 1
```

#### 3. Train Poisoned Model (Dynamic Noise Defense)
```bash
python3 train.py --Mixup 'Dynamic' --experiment-name 'MD-DYR-SH' --epochs 300 \
    --M 100 250 --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 2.0 --device "cuda:1" --exp 1 --distribution gaussian
```

Deploy Membership Inference Attack:
```bash
python3 privacy_risks_evaluations/metrics_based_MIA.py --Mixup 'Dynamic' \
    --experiment-name 'MD-DYR-SH' --epochs 300 --M 100 250 --distribution gaussian \
    --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 4.0 --device "cuda:0" --exp 1
```

#### 4. Train Poisoned Model (Static Noise Defense)
```bash
python3 train.py --Mixup 'Static' --experiment-name 'MD-DYR-SH' --epochs 300 \
    --M 100 250 --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 2.0 --device "cuda:0" --exp 1 --distribution gaussian
```

Deploy Membership Inference Attack:
```bash
python3 privacy_risks_evaluations/metrics_based_MIA.py --Mixup 'Dynamic' \
    --experiment-name 'MD-DYR-SH' --epochs 300 --M 100 250 --distribution gaussian \
    --reg-term 1.0 --model_name "resnet18" --dataset "CIFAR10" \
    --noise_level 4.0 --device "cuda:0" --exp 1
```

## Project Structure

```
.
├── train.py                    # Main training script
├── train_lira.py              # LIRA training implementation
├── NDSS_2025_facilitate_PMIA.py # Attack evaluation
├── privacy_risks_evaluations/  # Privacy evaluation tools
├── models_set/                 # Model architectures
└── utils.py                   # Utility functions
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{PMIANLL2025,
    title={PMIANLL: Defending against Poisoning Membership Inference Attacks with Noisy Label Learning},
    author={Jian Li, Xiaochun Yang, Wanlun Ma et al.},
    booktitle={Knowledge Based Systems},
    year={2025}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

We thank the authors of previous works on membership inference attacks and noisy label learning for their valuable insights and contributions to the field.

## References

[1] Arazo E, Ortego D, Albert P, et al. Unsupervised label noise modeling and loss correction[C]//International conference on machine learning. PMLR, 2019: 312-321.