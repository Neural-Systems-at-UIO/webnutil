import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_file
from nutil import Nutil
import os
import tempfile
from werkzeug.utils import secure_filename
import zipfile
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload
# for segmented files this checks out, why? due to compression
app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "json",
    "waln",
    "nrrd",
    "csv",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


# only one template for now, main use is to allow ui access to the functions.


@app.route("/process", methods=["POST"])
def process():
    try:
        # Get form data
        segmentation_files = request.files.getlist("segmentation_files")
        alignment_file = request.files["alignment_file"]
        atlas_file = request.files["atlas_file"]
        label_file = request.files["label_file"]

        colour_r = int(request.form.get("colour_r", 0))
        colour_g = int(request.form.get("colour_g", 0))
        colour_b = int(request.form.get("colour_b", 255))

        object_cutoff = int(request.form.get("object_cutoff", 0))
        use_flat = request.form.get("use_flat", "false") == "true"

        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        seg_dir = os.path.join(temp_dir, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)

        # Save segmentation files
        for file in segmentation_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(seg_dir, filename))

        # Save alignment file
        alignment_path = os.path.join(
            temp_dir, secure_filename(alignment_file.filename)
        )
        alignment_file.save(alignment_path)

        # Save atlas file
        atlas_path = os.path.join(temp_dir, secure_filename(atlas_file.filename))
        atlas_file.save(atlas_path)

        # Save label file
        label_path = os.path.join(temp_dir, secure_filename(label_file.filename))
        label_file.save(label_path)

        # Process with Nutil
        nt = Nutil(
            segmentation_folder=seg_dir,
            alignment_json=alignment_path,
            colour=[
                colour_b,
                colour_g,
                colour_r,
            ],  # BGR format - this is fixed for the browser ui
            atlas_path=atlas_path,
            label_path=label_path,
        )

        nt.get_coordinates(object_cutoff=object_cutoff, use_flat=use_flat)
        nt.quantify_coordinates()

        # Save analysis
        output_dir = os.path.join(temp_dir, "output")
        nt.save_analysis(output_dir)

        # Create zip file with results
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        memory_file.seek(0)

        # initiate this as a download immediately
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name="nutil_results.zip",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
