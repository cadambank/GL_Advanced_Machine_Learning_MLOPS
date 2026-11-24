# SuperKart Sales Forecasting — MLOps Pipeline

End-to-end MLOps pipeline that forecasts `Product_Store_Sales_Total` for SuperKart and deploys it with CI/CD.

| Stage | Script | Output |
|---|---|---|
| Data registration | `superkart_project/model_building/data_register.py` | [HF dataset](https://huggingface.co/datasets/cadambank/superkart-sales-data) |
| Data preparation | `superkart_project/model_building/prep.py` | train/test splits on the HF dataset |
| Training & registration | `superkart_project/model_building/train.py` | [HF model](https://huggingface.co/cadambank/superkart-sales-model) |
| Deployment | `superkart_project/hosting/hosting.py` + `superkart_project/deployment/` | [HF Space](https://huggingface.co/spaces/cadambank/superkart-sales-forecast) |
| Automation | `.github/workflows/pipeline.yml` | runs on every push to `main` |

```
.
├── .github/workflows/pipeline.yml
├── requirements.txt
└── superkart_project/
    ├── data/SuperKart.csv
    ├── model_building/  (data_register.py, prep.py, train.py, metrics.json)
    ├── deployment/      (Dockerfile, app.py, requirements.txt, README.md)
    └── hosting/         (hosting.py)
```
