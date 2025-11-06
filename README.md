Brain Tumor MRI Classification with ViT and GradioThis project uses a Vision Transformer (ViT) model, fine-tuned with PyTorch, to classify brain tumors from MRI scans. It classifies images into four categories: glioma, meningioma, no tumor, and pituitary.The project includes the original training notebook and a standalone web application built with Gradio to provide an interactive demo for the model.



Features4-Class Classification: Distinguishes between Glioma, Meningioma, Pituitary tumors, and No Tumor.Modern Architecture: Built on the powerful Vision Transformer (ViT) architecture (google/vit-base-patch16-224) for high accuracy.
Interactive UI: A simple Gradio web app allows you to upload an image and instantly get a prediction with confidence scores.
Complete Workflow: Includes the final trained models and the MRI_MODEL_RESNET50_VIT.ipynb notebook detailing the entire process from data loading and augmentation to model training and evaluation.
Technology UsedPyTorch: For model training and inference.Hugging Face transformers: For the pre-trained ViT model.
Gradio: For building and serving the interactive web demo.Scikit-learn: For calculating metrics (accuracy, classification report).
Jupyter Notebook: For experimentation and training.
Git LFS: (Git Large File Storage) To handle the large .pth model files.Project StructureHere is an overview of the key files in this repository:.
├── .gitattributes         # Configures Git LFS to track .pth files
├── demo/
│   └── gradio_app_demo.png  # Screenshot for this README
├── gradio_app.py          # The main Gradio application file
├── MRI_MODEL_RESNET50_VIT.ipynb # The original notebook for training and experimentation
├── vit_base_best.pth      # The trained ViT model weights (used by the app)
├── resnet50_best.pth    # The trained Res-Net-50 model weights (for comparison)
├── requirements.txt       # A list of all Python dependencies
└── README.md              # This file

How to Run This Project LocallyYou can easily run the Gradio demo on your own machine.1. PrerequisitesPython 3.8+Git and Git LFS (Git Large File Storage)2. Clone the RepositoryFirst, you must have git-lfs installed. Then, clone the repository. Using git lfs clone is the best way to ensure the large model files are downloaded correctly.# Install git-lfs (you only need to do this once per machine)
# On macOS: brew install git-lfs
# On Windows: git lfs install
# On Linux: sudo apt-get install git-lfs
# git lfs install

# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

3. Set Up a Virtual EnvironmentIt's highly recommended to use a virtual environment.# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

4. Install DependenciesInstall all the required Python libraries from the requirements.txt file.pip install -r requirements.txt

5. Run the Gradio AppYou're all set! Just run the gradio_app.py script.python gradio_app.py

This will start a local web server. Open the URL printed in your terminal (usually http://127.0.0.1:7860) in your browser to use the app!