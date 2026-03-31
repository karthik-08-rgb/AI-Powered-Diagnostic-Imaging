import os
import sqlite3
from flask import Flask, request, render_template_string, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from fpdf import FPDF
import numpy as np

# -------------------- Setup Flask --------------------
app = Flask(__name__)
app.secret_key = "narayan"
UPLOAD_FOLDER = "/content/drive/MyDrive/MedicalData/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -------------------- Database Setup --------------------
DB_PATH = "/content/drive/MyDrive/MedicalData/medical_records.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, age INTEGER, gender TEXT, email TEXT,
    modality TEXT, disease TEXT, confidence REAL,
    treatment TEXT, image_path TEXT
)
""")
conn.commit()

# -------------------- Load Models --------------------
XRAY_MODEL = load_model("/content/drive/MyDrive/MedicalData/xray_model.h5")
MRI_MODEL = load_model("/content/drive/MyDrive/MedicalData/mri_model.h5")
MODALITY_MODEL = load_model("/content/drive/MyDrive/MedicalData/modality_model1.h5")

XRAY_CLASSES = ["NORMAL", "PNEUMONIA"]
MRI_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# -------------------- Disease Info --------------------
DIAGNOSIS_MAP = {
    "xray": {
        "NORMAL": {
            "Disease": "Normal Chest X-ray",
            "Treatment": "No acute pulmonary findings detected. Continue routine care and follow-up if needed."
        },
        "PNEUMONIA": {
            "Disease": "Pneumonia",
            "Treatment": "Suspected lobar pneumonia. Initiate broad-spectrum antibiotics and supportive care. "
                         "Monitor patient condition closely, check oxygen saturation, and schedule follow-up imaging. "
                         "Urgent medical consultation is recommended if symptoms worsen."
        }
    },
    "mri": {
        "glioma": {
            "Disease": "Glioma",
            "Treatment": "Suspected high-grade glioma. Immediate neurosurgical consultation is recommended to evaluate "
                         "for biopsy or resection. Post-surgical management may include radiation therapy and chemotherapy. "
                         "Close monitoring with follow-up MRI scans is advised. Multidisciplinary team involvement including "
                         "neuro-oncology, neurosurgery, and supportive care is essential."
        },
        "meningioma": {
            "Disease": "Meningioma",
            "Treatment": "Probable meningioma. Referral to neurosurgery for evaluation is recommended. Depending on tumor size, "
                         "location, and symptoms, management may involve surgical resection or active monitoring. "
                         "Periodic imaging follow-up is advised to monitor growth."
        },
        "notumor": {
            "Disease": "Normal MRI",
            "Treatment": "No tumor detected. Routine monitoring recommended. Maintain regular follow-ups and report any new "
                         "neurological symptoms promptly."
        },
        "pituitary": {
            "Disease": "Pituitary Tumor",
            "Treatment": "Pituitary adenoma suspected. Endocrinology referral for hormone evaluation is advised. "
                         "Neurosurgical consultation may be necessary depending on size and symptoms. Consider MRI follow-up "
                         "to monitor tumor growth."
        }
    }
}

# -------------------- Helper Functions --------------------
import numpy as np
from PIL import Image

def preprocess_image(file_path, target_size):
    """Resize and normalize image for model input."""
    img = load_img(file_path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def detect_modality(file_path):
    """Predict whether an image is X-ray, MRI, or Other."""
    preds = MODALITY_MODEL.predict(preprocess_image(file_path, (128, 128)))[0]
    classes = ["xray", "mri", "other"]
    label = classes[np.argmax(preds)]
    confidence = float(np.max(preds))
    print(f"[DEBUG] Detected Modality: {label} (Confidence: {confidence:.2f}) for {os.path.basename(file_path)}")
    return label, confidence


def validate_modality(file_path, expected_modality):
    """Validate uploaded medical image type and reject invalid ones."""
    try:
        Image.open(file_path).verify()
    except Exception:
        return False, "The uploaded file is not a valid image."

    # Step 1: Model-based modality detection
    detected, confidence = detect_modality(file_path)

    # Step 2: Reject low-confidence or non-medical predictions
    if detected == "other" or confidence < 0.75:
        return False, "The uploaded image does not appear to be an X-ray or MRI."

    # Step 3: Reject mismatched modality
    if detected != expected_modality:
        return False, f"You selected {expected_modality.upper()}, but the uploaded image looks like {detected.upper()}."

    # Step 4: Validation passed
    return True, "OK"


def predict(file_path, modality):
    """Predict the disease class for X-ray or MRI after validation."""
    if modality == "xray":
        preds = XRAY_MODEL.predict(preprocess_image(file_path, (224, 224)))
        idx = np.argmax(preds[0])
        disease = XRAY_CLASSES[idx]
        confidence = float(preds[0][idx])
        print(f"[DEBUG] X-ray Prediction: {disease} ({confidence:.2f})")
        return disease, confidence

    elif modality == "mri":
        preds = MRI_MODEL.predict(preprocess_image(file_path, (224, 224)))
        idx = np.argmax(preds[0])
        disease = MRI_CLASSES[idx]
        confidence = float(preds[0][idx])
        print(f"[DEBUG] MRI Prediction: {disease} ({confidence:.2f})")
        return disease, confidence

    print("[DEBUG] Unknown modality provided to predict()")
    return "Unknown", 0.0



def save_record(name, age, gender, email, modality, disease, confidence, treatment, image_path):
    cursor.execute("""
    INSERT INTO records (name, age, gender, email, modality, disease, confidence, treatment, image_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, age, gender, email, modality, disease, confidence, treatment, image_path))
    conn.commit()
    return cursor.lastrowid



def generate_pdf(record_id, name, age, gender, email, modality, disease, confidence, treatment, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Background
    pdf.set_fill_color(255, 255, 255)  # White
    pdf.rect(0, 0, 210, 297, 'F')

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(33, 37, 41)  # Dark gray
    pdf.cell(0, 15, "Medical Diagnosis Report", ln=True, align="C")
    pdf.ln(10)

    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Name: {name}\nAge: {age}\nGender: {gender}\nEmail: {email}")
    pdf.ln(5)

    # Diagnosis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnosis Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Modality: {modality.upper()}\nDisease: {disease}\nConfidence: {confidence * 100:.2f}%")
    pdf.ln(5)

    # Treatment
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Treatment Plan", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, treatment)
    pdf.ln(10)

    # Image
    if os.path.exists(image_path):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Diagnostic Image", ln=True)
        pdf.image(image_path, x=65, w=80)
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "Figure: Uploaded diagnostic image", ln=True, align="C")

    # Save PDF
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"report_{record_id}.pdf")
    pdf.output(output_path)
import os, shutil

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

# Copy background images (adjust paths as needed)
shutil.copy('/content/drive/MyDrive/MedicalData/photo.png', 'static/photo.png')
shutil.copy('/content/drive/MyDrive/MedicalData/image.png', 'static/image.png')

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medical Diagnosis</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: url("{{ url_for('static', filename='photo.png') }}") no-repeat center center fixed;
      background-size: cover;
      color: #d5edff;
      font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
      min-height: 100vh;
    }

    /* 📱 Change background for mobile devices */
    @media only screen and (max-width: 768px) {
      body {
        background: url("{{ url_for('static', filename='image.png') }}") no-repeat center center fixed;
        background-size: cover;
      }
    }

    .card {
      background: rgba(12, 26, 44, 0.40);
      backdrop-filter: blur(12px);
      border-radius: 20px;
      box-shadow: 0 4px 32px rgba(0, 80, 180, 0.5);
      border: 1.5px solid #3ad1e8;
      color: #d5edff;
    }
    .card-title {
      font-size: 2.1rem;
      font-weight: bold;
      letter-spacing: 1.5px;
      text-shadow: 0 2px 10px #18e6ff8f;
    }
    .form-label {
      color: #75d7f0;
    }

    /* Updated input & dropdown styling for clarity */
    .form-control, .form-select {
      background: rgba(255, 255, 255, 0.15);
      border: 1.5px solid #28b8ef;
      color: #ffffff;
      border-radius: 10px;
      box-shadow: 0 0 8px #28b8ef52;
      appearance: none;
    }

    /* Ensures dropdown options (X-ray / MRI) are readable */
    .form-select option {
      background-color: #0b233d;
      color: #ffffff;
      padding: 10px;
    }

    /* Highlight on hover inside dropdown */
    .form-select option:hover {
      background-color: #18e6ff;
      color: #002d46;
      font-weight: 600;
    }

    .form-control::placeholder {
      color: #98bccc;
    }
    .form-control:focus, .form-select:focus {
      background: rgba(40, 184, 239, 0.13);
      border-color: #18e6ff;
      box-shadow: 0 0 20px #18e6ff2e;
      color: #fff;
      outline: none;
    }

    .btn {
      transition: all 0.3s ease;
      font-weight: 500;
      border-radius: 10px;
    }
    .btn-primary {
      background: linear-gradient(90deg, #0bebf2 30%, #18e6ff 70%);
      box-shadow: 0 2px 10px #18e6ff80;
      border: none;
      color: #003550;
    }
    .btn-success {
      background: linear-gradient(90deg, #44efb2 0%, #0bebf2 100%);
      box-shadow: 0 2px 10px #0bebf280;
      border: none;
      color: #003550;
    }
    .btn-danger {
      background-color: #dc3545;
      border: none;
      color: #fff;
    }
    .btn-secondary {
      background: linear-gradient(90deg, #374f75 0%, #67a1cc 100%);
      border: none;
      color: #d5edff;
    }
    .btn:hover {
      transform: scale(1.05);
      opacity: 0.9;
    }
    .alert {
      border-radius: 16px;
      background: rgba(40, 184, 239, 0.18);
      color: #fff;
      border: 1.5px solid #18e6ff80;
    }
    .img-fluid {
      border-radius: 14px;
      box-shadow: 0 0 18px #18e6ff46;
      border: 1.5px solid #18e6ff63;
    }
</style>

</head>
<body>
  <div class="container mt-5">
    <div class="card shadow p-4">
      <h2 class="card-title text-center mb-4"> Medical Image Diagnosis</h2>
      <form method="post" enctype="multipart/form-data">
        <div class="row">
          <div class="col-md-6 mb-3">
            <label class="form-label">Full Name</label>
            <input type="text" name="name" class="form-control" required>
          </div>
          <div class="col-md-3 mb-3">
            <label class="form-label">Age</label>
            <input type="number" name="age" class="form-control" required>
          </div>
          <div class="col-md-3 mb-3">
            <label class="form-label">Gender</label>
            <select name="gender" class="form-select" required>
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Email</label>
          <input type="email" name="email" class="form-control" required>
        </div>
        <div class="mb-3">
          <label class="form-label">Select Image Type</label>
          <select class="form-select" name="modality" required>
            <option value="xray">X-ray</option>
            <option value="mri">MRI</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Upload Image</label>
          <input type="file" class="form-control" name="file" accept="image/*" required>
        </div>
        <div class="d-grid">
          <button type="submit" class="btn btn-primary btn-lg">Diagnose</button>
        </div>
      </form>

       {% if error %}
       <hr>
      <div class="alert alert-danger mt-3">{{ error|safe }}</div>
       {% endif %}


      {% if disease %}
      <hr>
      <div class="alert alert-success mt-3">
        <h4>Disease: {{ disease }}</h4>
        <p><strong>Treatment:</strong></p>
        {% for line in treatment.split('. ') %}<p>{{ line }}.</p>{% endfor %}
        <p><strong>Confidence:</strong> {{ confidence*100|round(2) }}%</p>
        <img src="{{ url_for('uploaded_file', filename=image_filename) }}" class="img-fluid rounded shadow mt-3" style="max-width:400px;">
        <div class="mt-3">
          <a href="{{ url_for('download_report', record_id=record_id) }}" class="btn btn-success">Download Report</a>
        </div>
      </div>
      {% endif %}

      <div class="mt-3 text-center">
        <a href="{{ url_for('view_records') }}" target="_blank" class="btn btn-secondary mt-2">View All Records</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    from flask import render_template_string
    disease = treatment = confidence = error = image_filename = None
    record_id = None

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        email = request.form["email"]
        modality = request.form["modality"]
        f = request.files.get("file")

        if not f:
            error = "No file uploaded."
            return render_template_string(HTML_PAGE, error=error)

        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)

        valid, msg = validate_modality(path, modality)
        if not valid:
            error = msg
            os.remove(path)
            return render_template_string(HTML_PAGE, error=error)

        result_key, confidence = predict(path, modality)
        result = DIAGNOSIS_MAP[modality].get(result_key)
        if not result:
            error = "Unknown diagnosis."
            return render_template_string(HTML_PAGE, error=error)

        disease = result["Disease"]
        treatment = result["Treatment"]
        image_filename = filename

        record_id = save_record(name, age, gender, email, modality, disease, confidence, treatment, path)
        generate_pdf(record_id, name, age, gender, email, modality, disease, confidence, treatment, path)

        # ✅ Save user email in session for secure access
        session["user_email"] = email

    return render_template_string(HTML_PAGE, disease=disease, treatment=treatment, confidence=confidence,
                                  error=error, image_filename=image_filename, record_id=record_id)

@app.route('/download/<int:record_id>')
def download_report(record_id):
    # ✅ Patient-specific download protection
    user_email = session.get("user_email")
    if not user_email:
        return "<h3 style='color:red;text-align:center;margin-top:50px;'>❌ Unauthorized. Please upload your scan again to download your report.</h3>"

    cursor.execute("SELECT email FROM records WHERE id=?", (record_id,))
    record = cursor.fetchone()
    if not record:
        return "<h3 style='color:red;text-align:center;margin-top:50px;'>❌ Report not found.</h3>"
    if record[0] != user_email:
        return "<h3 style='color:red;text-align:center;margin-top:50px;'>❌ Access denied. You are not authorized to download this report.</h3>"

    return send_from_directory(app.config["UPLOAD_FOLDER"], f"report_{record_id}.pdf", as_attachment=True)

# ✅ Admin authentication added
# ✅ Admin authentication (secure + persistent)
from flask import session, request

@app.before_request
def restrict_admin_access():
    # Only protect the /records route
    if request.endpoint == "view_records":
        # If already logged in, allow access
        if session.get("admin"):
            return

        # If POST request (trying to log in)
        if request.method == "POST":
            key = request.form.get("admin_key")
            if key == "narayan@123":  # ✅ Your private key here
                session["admin"] = True
                return
            else:
                return """
                <script>alert('❌ Invalid key. Access denied.');window.location.href='/records';</script>
                """

        # If not logged in and not posting a key, show the login form
        return """
        <form method='POST' style='max-width:400px;margin:100px auto;text-align:center;'>
            <h3> Admin Access Required</h3>
            <input type='password' name='admin_key' placeholder='Enter admin key' required class='form-control mb-3'>
            <button class='btn btn-primary w-100'>Login</button>
        </form>
        """


@app.route("/logout_admin")
def logout_admin():
    session.pop("admin", None)
    return "<h3 style='color:green;text-align:center;margin-top:50px;'>✅ Admin logged out successfully.</h3>"


@app.route('/delete/<int:record_id>')
def delete_record(record_id):
    cursor.execute("DELETE FROM records WHERE id=?", (record_id,))
    conn.commit()

    # Reorder all remaining records
    records = cursor.execute("SELECT * FROM records ORDER BY id").fetchall()
    cursor.execute("DELETE FROM records")
    for new_id, r in enumerate(records):
        cursor.execute("""
            INSERT INTO records (id, name, age, gender, email, modality, disease)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (new_id, r[1], r[2], r[3], r[4], r[5], r[6]))
    conn.commit()

    cursor.execute("DELETE FROM sqlite_sequence WHERE name='records'")
    conn.commit()

    return "<script>alert('Record deleted successfully');window.location.href='/records';</script>"

@app.route('/delete_all')
def delete_all_records():
    cursor.execute("DELETE FROM records")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='records'")  # reset AUTOINCREMENT
    conn.commit()
    return "<script>alert('All records deleted successfully');window.location.href='/records';</script>"

# -------------------- Updated /records Route --------------------
@app.route('/records', methods=["GET", "POST"])

def view_records():
    from flask import render_template_string
    records = cursor.execute(
        "SELECT id, name, age, gender, email, modality, disease FROM records ORDER BY id"
    ).fetchall()

    html = """
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    <div class="container mt-5">
      <div class="card p-4 shadow-lg" style="border-radius:15px; background-color:white;">
        <h2 class="text-primary mb-4 fw-bold text-center">Stored Medical Records</h2>
        <hr class="text-muted">

        <div class="d-flex justify-content-end mb-3">
          <a href='/delete_all' class='btn btn-danger shadow'
             onclick="return confirm(' Are you sure you want to delete ALL records?');">🗑 Delete All Records</a>
        </div>

        <div class="table-responsive">
          <table class="table table-hover align-middle shadow-sm rounded">
            <thead class="table-primary text-center text-white">
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Email</th>
                <th>Modality</th>
                <th>Disease</th>
                <th>ACTIONS</th>
              </tr>
            </thead>
            <tbody class="text-center">
    """

    # Add all records in table
    for idx, r in enumerate(records):
        html += f"""
              <tr class="{'' if idx % 2 == 0 else 'table-light'}">
                <td>{r[0]}</td>
                <td>{r[1]}</td>
                <td>{r[2]}</td>
                <td>{r[3]}</td>
                <td>{r[4]}</td>
                <td>{r[5]}</td>
                <td>{r[6]}</td>
                <td>
                  <a href='/delete/{r[0]}' class='btn btn-outline-danger btn-sm rounded-pill shadow-sm'
                     onclick="return confirm('Are you sure you want to delete this record?');">🗑 Delete</a>
                </td>
              </tr>
        """

    html += """
            </tbody>
          </table>
        </div>

        <div class="text-center mt-3">
          <a href='/' class='btn btn-link text-primary'>&larr; Back</a>
        </div>
      </div>
    </div>

    <style>
      body { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
      .table-hover tbody tr:hover { background-color: #e9f3ff; }
      .btn-outline-danger:hover { background-color: #dc3545; color: white; }
      .table thead th { font-weight: 600; }
      .card { border-radius: 20px; }
    </style>
    """
    return render_template_string(html)




# -------------------- Run App --------------------
from pyngrok import ngrok
port = 5000
public_url = ngrok.connect(port)
print(" Flask app running at:", public_url)
app.run(port=port)