# 🚀 MLOps Class: Image Classifier Deployment on GCP

This applied class project is the final MLOps deployment from the **Data Science Retreat** program. It demonstrates a robust, traceable pipeline for a **ResNet Image Classifier** (Fresh/Rotten Fruit classification).

⚠️ **Deployment Status:** The live application was successfully deployed on Google Cloud Platform (GCP) but has been deleted to manage costs. This repository maintains the complete, documented infrastructure for demonstrating pipeline and deployment expertise.

## ⚙️ Architecture and Key Technologies

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Serving Endpoint** | **FastAPI** | Provides a scalable, high-performance API for image prediction. |
| **Data Validation** | Pydantic | Ensures robust API request and response data integrity. |
| **Artifact Management** | **Weights & Biases (W&B)** | Dynamically fetches version-controlled model weights (`best_model.pth`) via the W&B API at container startup. |
| **Model** | PyTorch / ResNet | Optimized model loading and inference using `torch.inference_mode()`. |
| **Containerization** | Docker | Ensures a portable, consistent runtime environment for deployment. |
| **Cloud Target** | Google Cloud Platform (GCP) | Repository showcases configuration for services like Cloud Run or Compute Engine. |

## 📁 Repository Structure

The core application logic is contained within the `app/` directory, following industry standards for containerized microservices.

* `app/main.py`: Defines the FastAPI endpoint (`/predict`).
* `app/model.py`: **Crucial:** Handles artifact fetching, environment configuration, and PyTorch model loading from W&B.
* `Dockerfile`: Defines the build environment and startup command.
* `requirements.txt`: Specifies all dependencies, including `wandb` and `torch`.
