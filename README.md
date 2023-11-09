# Yet Another Image Classification CNN Experiment on BreakHis Dataset

A computer vision model for classifying breast cancer histopathology images, supports classification according to the following perspectives

- Classes
  - Benign (B)
  - Malignant (M)
- Types
  - Adenosis (A)
  - Fibroadenoma (F)
  - Phyllodes Tumor (PT)
  - Tubular Adenona (TA)
  - Carcinoma (DC)
  - Lobular Carcinoma (LC)
  - Mucinous Carcinoma (MC)
  - Papillary Carcinoma (PC)
- Magnifications
  - 40x
  - 100x
  - 200x
  - 400x

Description from [kaggle](https://www.kaggle.com/datasets/ambarish/breakhis):

> The dataset BreaKHis is divided into two main groups: benign tumors and malignant tumors. Histologically benign is a term referring to a lesion that does not match any criteria of malignancy – e.g., marked cellular atypia, mitosis, disruption of basement membranes, metastasize, etc. Normally, benign tumors are relatively “innocents”, presents slow growing and remains localized. Malignant tumor is a synonym for cancer: lesion can invade and destroy adjacent structures (locally invasive) and spread to distant sites (metastasize) to cause death.
>
> In current version, samples present in dataset were collected by SOB method, also named partial mastectomy or excisional biopsy. This type of procedure, compared to any methods of needle biopsy, removes the larger size of tissue sample and is done in a hospital with general anesthetic.
>
> Both breast tumors benign and malignant can be sorted into different types based on the way the tumoral cells look under the microscope. Various types/subtypes of breast tumors can have different prognoses and treatment implications. The dataset currently contains four histological distinct types of benign breast tumors: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA); and four malignant tumors (breast cancer): carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC).

The repository utilizes several different models pretrained on ImageNet and finetune them on BreakHis.

# Get Started

This repository basically contains 3 part:

1. Split the BreakHis dataset
1. Load and preprocess the dataset
1. Train and evaluate classifiers on the dataset

## Dependencies

Follow the instructions below to build the environment

1. Create and activate a new environment using conda

   ```bash
   conda create -n breakhis python=3.9
   conda activate breakhis
   ```

2. Clone the repository, navigate into it and install dependencies using `requirements.txt`

   ```bash
   git clone https://github.com/lizbaka/BreakHis-exp.git
   cd BreakHis-exp
   pip install -r requirements.txt
   ```

## Prepare dataset

3. Make a `dataset` directory in the root of the project, then download the BreaKHis dataset from [kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) and place it into the `dataset` folder. The directory tree should look like this:

    ```
    dataset/
    └── BreaKHis_v1/
        ├── Folds.csv
        └── BreaKHis_v1/
            └── histology_slides/
                └── breast/
                    ├── benign/
                    │   └── ...
                    └── malignant/
                        └── ...
    ```

4. Split the dataset using `mysplit.py`.

    ```bash
    python mysplit.py
    ```

    And `dataset/BreaKHis_v1/mysplit.csv` should be generated:

    ```
    mag_grp,grp,tumor_class,tumor_type,path
    40,train,B,A,BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-004.png
    ...
    ```

    It will split the dataset into three parts: train, dev and test set, with the ratio of 7:2:1

    > According to [[1]](#Reference), images corresponding to the patient with ID:13412 are replicated in two malignant sub-categories: (DC) and (LC), which may confuse the classifier. We filtered these images in the generated split file.

    See `BreaKHis_generate` class in `datasets.py` for more information.

## Fire up training!

5. Train (or evaluate only) classifiers with `main.py`. Here is an example:

    ```bash
    python main.py \
        --task binary \
        --net ResNet50 \
        --output_dir ResNet50 \
        --batch_size 32 \
        --epoch 20 \
        --lr 1e-4 \
        --best_metric acc \
        --da \
        --resume
    ```

    Supported args are:

    | Args          | Expected value(s)                                            | Note                                                         |
    | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | `task`        | binary, subtype or magnification                             | Correspond to Classes, Types and Magnifications              |
    | `net`         | `ResNet50`, `ResNet152`, `DenseNet121`, `DenseNet201`, `VGG11`, `VGG19_bn`, `ResNeXt_101_32x8d` | You can add your own. See `networks.py`                      |
    | `output_dir`  | path to the output dir                                       | Where best and last checkpoints, tfevents, configs and test results are stored |
    | `batch_size`  | integer (optional)                                           | `32` if not specified                                        |
    | `epoch`       | integer (optional)                                           | `20` if not specified                                        |
    | `lr`          | float (optional)                                             | Learning rate, `1e-4`  if not specified                      |
    | `mag`         | `40`, `100`, `200`, `400` (optional)                         | Use a subset of the dataset with certain magnification when specified, or use the whole dataset instead |
    | `best_metric` | `acc`, `auroc`, `f1`, `precision`, `recall` (optional)       | Metrics are computed with macro average. use `auroc` if not specified |
    | `ckpt`        | path to a checkpoint file                                    | Which checkpoint to evaluate or to continue training on      |
    | `resume`      | *switch* (optional)                                          | Whether to continue training on `last.pth` when existed      |
    | `da`          | *switch* (optional)                                          | Whether to use data augmentation (Random Hflip and Vflip)    |
    | `eval`        | *switch* (optional)                                          | Whether to skip training. `ckpt` or `output_dir` contains `besk.pth` should be specified |

6. Visualization (optional)

    Visualize the training procedure using `Tensorboard`. After installing it, run:

    ```bash
    tensorboard --logdir path/to/output_dir --port 6006
    ```

    and visit `http://localhost:6006` to see the visualized training procedure.

    Monitored metrics include loss, accuracy, auroc, f1, precision, recall on dev set and loss on train set.

# Reference Results

We trained several different neural networks using this project. 

Training are done on a single RTX 3090 24G graphics card. 

Average inference time is evaluated on a mobile RTX 2060 6G GPU with `batch size` of `4`, obtained from averaging time elapsed on evaluation on test for 5 times.

We use Cross Entropy as our loss criterion, AdamW (with 0.01 weight decay) as our optimizer and auroc as the best metric. Learning rate drops to `0.8` times the original after each epoch.

Here are the results we gained:

On binary task:

- config: 

    | model             | batch size | learning rate | epoch |
    | ----------------- | ---------- | ------------- | ----- |
    | DenseNet121       | 16         | 1e-5          | 20    |
    | DenseNet201       | 16         | 1e-5          | 20    |
    | ResNet152         | 16         | 1e-5          | 20    |
    | ResNet50          | 32         | 1e-4          | 20    |
    | ResNet50 w/DA     | 32         | 1e-4          | 20    |
    | ResNeXt_101_32x8d | 8          | 1e-5          | 20    |
    | VGG11             | 32         | 1e-4          | 20    |
    | VGG19_bn          | 16         | 1e-5          | 20    |

- results: 

    | model             | acc_macro | acc_micro | P_macro   | P_micro   | R_macro   | R_micro   | F1_macro  | F1_micro  | auroc     | inf_time  |
    | ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
    | DenseNet121       | 0.973     | 0.977     | 0.973     | 0.977     | 0.973     | 0.977     | 0.973     | 0.977     | 0.993     | 50.07     |
    | DenseNet201       | 0.981     | 0.980     | 0.975     | 0.980     | 0.981     | 0.980     | 0.978     | 0.980     | 0.993     | 59.55     |
    | ResNet152         | 0.968     | 0.969     | 0.962     | 0.969     | 0.968     | 0.969     | 0.965     | 0.969     | 0.993     | 67.12     |
    | ResNet50          | 0.990     | 0.990     | 0.986     | 0.990     | 0.990     | 0.990     | 0.988     | 0.990     | 0.995     | **45.29** |
    | ResNet50 w/DA     | **0.995** | **0.995** | **0.993** | **0.995** | **0.995** | **0.995** | **0.994** | **0.995** | **0.997** | **45.29** |
    | ResNeXt_101_32x8d | 0.973     | 0.970     | 0.961     | 0.970     | 0.973     | 0.970     | 0.966     | 0.970     | 0.994     | 85.50     |
    | VGG11             | 0.969     | 0.973     | 0.968     | 0.973     | 0.969     | 0.973     | 0.969     | 0.973     | 0.985     | 51.27     |
    | VGG19_bn          | 0.972     | 0.975     | 0.971     | 0.975     | 0.972     | 0.975     | 0.972     | 0.975     | 0.992     | 77.22     |

On subtype task:

- config:

    | model               | batch size | learning rate | epoch |
    | ------------------- | ---------- | ------------- | ----- |
    | DenseNet121         | 16         | 1e-4          | 20    |
    | DenseNet201         | 16         | 1e-4          | 20    |
    | DenseNet201 w/DA    | 16         | 1e-4          | 20    |
    | DenseNet201 w/noise | 16         | 1e-4          | 20    |
    | ResNet152           | 16         | 1e-4          | 20    |
    | ResNet50            | 32         | 1e-4          | 20    |
    | ResNet50-da         | 32         | 1e-4          | 20    |
    | ResNeXt_101_32x8d   | 8          | 1e-4          | 20    |
    | VGG11               | 32         | 1e-4          | 20    |
    | VGG19_bn            | 16         | 1e-4          | 20    |

- results:

    | model               | acc_macro | acc_micro | P_macro   | P_micro   | R_macro   | R_micro   | F1_macro  | F1_micro  | auroc     | inf_time  |
    | ------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
    | DenseNet121         | 0.965     | 0.969     | 0.957     | 0.969     | 0.965     | 0.969     | 0.961     | 0.969     | 0.991     | 50.86     |
    | DenseNet201         | **0.965** | **0.975** | **0.974** | **0.975** | **0.965** | **0.975** | **0.969** | **0.975** | **0.992** | 60.71     |
    | DenseNet201 w/DA    | 0.962     | 0.973     | 0.968     | 0.973     | 0.962     | 0.973     | 0.965     | 0.973     | 0.992     | 60.71     |
    | DenseNet201 w/noise | 0.918     | 0.928     | 0.926     | 0.928     | 0.918     | 0.928     | 0.921     | 0.928     | 0.983     | 60.71     |
    | ResNet152           | 0.959     | 0.966     | 0.959     | 0.966     | 0.959     | 0.966     | 0.959     | 0.966     | 0.992     | 66.16     |
    | ResNet50            | 0.960     | 0.967     | 0.957     | 0.967     | 0.960     | 0.967     | 0.958     | 0.967     | 0.990     | **46.83** |
    | ResNet50 w/da       | 0.954     | 0.964     | 0.951     | 0.964     | 0.954     | 0.964     | 0.952     | 0.964     | 0.994     | **46.83** |
    | ResNeXt_101_32x8d   | 0.945     | 0.945     | 0.934     | 0.945     | 0.945     | 0.945     | 0.938     | 0.945     | 0.982     | 84.31     |
    | VGG11               | 0.910     | 0.931     | 0.917     | 0.931     | 0.910     | 0.931     | 0.913     | 0.931     | 0.973     | 51.50     |
    | VGG19_bn            | 0.931     | 0.941     | 0.924     | 0.941     | 0.931     | 0.941     | 0.926     | 0.941     | 0.980     | 76.66     |

# Reference

[1] Benhammou, Y. *et al.* (2020) ‘Breakhis based breast cancer automatic diagnosis using Deep Learning: Taxonomy, survey and insights’, *Neurocomputing*, 375, pp. 9–24. doi:10.1016/j.neucom.2019.09.044. 
