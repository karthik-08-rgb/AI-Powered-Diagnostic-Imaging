# AI-Powered-Diagnostic-Imaging
🚀 Overview
This project is an AI-based medical imaging system that analyzes X-ray and MRI scans to detect diseases automatically. It leverages deep learning models to assist healthcare professionals in making faster and more accurate diagnoses.
________________________________________
🎯 Features
•	📤 Upload medical images (X-ray / MRI) 
•	🧠 Automatic disease detection using AI models 
•	📊 Confidence score for predictions 
•	📄 Auto-generated diagnostic PDF reports 
•	🌐 User-friendly web interface using Flask 
•	💾 Stores patient data and results in database 
________________________________________
🛠️ Tech Stack
•	Programming Language: Python 
•	Framework: Flask 
•	Deep Learning: TensorFlow / Keras 
•	Models Used: DenseNet121, VGG16 
•	Libraries: NumPy, Pillow 
•	Database: SQLite 
•	Deployment: Ngrok (for public access) 
________________________________________
🧪 Model Details
🔹 X-ray Classification
•	Model: DenseNet121 
•	Classes: Normal, Pneumonia 
🔹 MRI Classification
•	Model: VGG16 
•	Classes: Glioma, Meningioma, Pituitary, No Tumor 
🔹 Modality Detection
•	Custom CNN to detect whether input is X-ray or MRI 
________________________________________
⚙️ How It Works
1.	User uploads an image (X-ray or MRI) 
2.	System validates image and detects modality 
3.	Image is preprocessed (resize, normalization) 
4.	Appropriate deep learning model is applied 
5.	Disease prediction + confidence score generated 
6.	Results displayed on UI + stored in database 
7.	PDF report generated for download 
________________________________________
📂 Project Structure
AI-Diagnostic-Imaging/
│── app.py                  # Flask application
│── train_xray.py          # X-ray model training
│── train_mri.py           # MRI model training
│── modality_detection.py  # CNN for modality detection
│── models/                # Saved .h5 models
│── static/uploads/        # Uploaded images
│── templates/             # HTML templates
│── database.db            # SQLite database
│── report/                # Generated PDF reports
________________________________________
▶️ Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application
python app.py
4. Access in Browser
http://127.0.0.1:5000/
________________________________________
📊 Results
•	Achieved high accuracy in disease classification 
•	Reliable predictions with confidence scoring 
•	Reduced manual effort in medical image analysis 
________________________________________
📌 Applications
•	🏥 Hospitals and diagnostic centers 
•	📚 Medical education and training 
•	🌍 Telemedicine platforms 
•	🔬 Research in medical imaging


🔬 Research in medical imaging
