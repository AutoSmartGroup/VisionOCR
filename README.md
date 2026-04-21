# Auto-Correct OCR: A Novel Method for Enhancing Character Recognition Accuracy through Error Correction

## introduction
This paper proposes Auto-Correct OCR framework, which consists of a training-free basic recognition module and a domain-adaptive post-processing module named Structure-Aware Correction (SAC).

This warehouse opens the experimental data used in this paper and the core code of the post-processing methods of all scenarios mentioned in the article, including two algorithmic frameworks, SAC for Fixed-Length strings (SACFL) and SAC for Variable-Length strings (SACVL).

We have preliminarily extracted the image data of the two scenes using the paddepaddle OCR engine, and saved the recognition results to two files, result_with_fixstr_fixed.xlsx and ppocr_resultfl_cleaned_with_noise.xlsx. The details of these two documents are as follows,  two files correspond to the brewery scene and the automotive parts factory scene respectively:
### Brewery Scene
<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Field</th><th>Type</th><th>Description</th></tr>
<tr><td>date</td><td>String</td><td>Date to which the sample belongs</td></tr>
<tr><td>pic</td><td>String</td><td>Filename of the sample image data</td></tr>
<tr><td>pse</td><td>String</td><td>Initially extracted string record</td></tr>
<tr><td>Identify</td><td>Bool</td><td>Whether it is completely correct (label field)</td></tr>
<tr><td>Fixstr</td><td>String</td><td>Fully correct string (label field)</td></tr>
</table>

### Automotive Parts Factory Scene

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Field</th><th>Type</th><th>Description</th></tr>
<tr><td>cinvaddcode</td><td>String</td><td>Inventory ID; filename without suffix</td></tr>
<tr><td>cinvname</td><td>String</td><td>Inventory name</td></tr>
<tr><td>cinvstd</td><td>String</td><td>Specification / model (label field)</td></tr>
<tr><td>iinvrcost</td><td>Float</td><td>Inventory receipt cost; not used in this study</td></tr>
<tr><td>inventorycode</td><td>String</td><td>Inventory code (label field)</td></tr>
<tr><td>ppocrstr1</td><td>String</td><td>Initially extracted inventory code string</td></tr>
<tr><td>ppocrstr2</td><td>String</td><td>Initially extracted specification/model string</td></tr>
<tr><td>ppocrstr3</td><td>String</td><td>Initially extracted specification/model string (augmented)</td></tr>
</table>

## Requirements
#### catboost==1.2.10
#### numpy==1.23.0
#### pandas==1.2.5
#### python_Levenshtein==0.27.3
#### scikit_learn==1.8.0
#### torch==2.5.1+cu121

## Directory Structure
```
├── SACFL     #SACFL experimental script and core algorithm code
│   ├── cross_time_validation_experiment.py  
│   ├── ppocr_resultFL_cleaned_with_noise.xlsx
│   ├── result_with_fixstr_fixed.xlsx
│   └── structure_aware_corrector.py   #algorithm code
│
├── SACVL     #SACVL experimental script and core algorithm code
│   ├── baselines
│   │   └── rbp
│   ├── checkpoints
│   ├── checkpoints_final
│   ├── logs_structure_aware
│   ├── train_figs
│   ├── vocabularies_dual
│   ├── dual_vocab_builder.py
│   ├── ppocr_resultFL_cleaned_with_noise.xlsx
│   ├── structure_aware_corrector_v5.py  #algorithm code
│   └── train_structure_corrector.ipynb  #train and test script
├── imgdata_Auto_parts_factory
│   ├── BCQ3270000001.png
│   ├── BCQ3270000002.png
│   └── ...
├── imgdata_Brewery
│   ├── may
│   │   ├── 10002.jpg
│   │   ├── 10005.jpg
│   │   └── ...
│   └── sept
│       ├── IMG_20240918_081229_782.png
│       ├── IMG_20240918_081230_913.png
│       └── ...
├── README.md
└── requirements.txt
```

## Dataset


Due to the large size of the dataset (approximately 6GB), it is not included in this repository. Please download it from the link below:

### 🔗 Download Link

Baidu Netdisk:
 https://pan.baidu.com/s/1nMapMPpCWdsg0AdADt5_pQ?pwd=1234

Extraction Code: 1234

---

### ⚠️ Notes

* The repository already includes character record files that were preliminarily extracted using open-source OCR engines.
* Reproducing this extraction step is not recommended, as it is unnecessary and time-consuming.
* Please ensure the dataset structure matches the project requirements.
* Check data paths in the configuration or source code if any issues occur.

---

If the link is unavailable or you encounter any problems, please contact the project maintainer.
