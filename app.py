import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from main import process_image

# Flask web server for upload, segmentation processing, heatmap generation, and file download.
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Use absolute paths based on this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'output')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}


def build_color_intensity_heatmap(image_bgr, alpha=0.45):
    """Create a visual heatmap where high saturation/brightness regions are emphasized."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    # Blend saturation and brightness so vivid + bright regions stand out.
    intensity = cv2.addWeighted(saturation, 0.7, value, 0.3, 0)
    intensity_u8 = np.clip(intensity, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(intensity_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

    return overlay, intensity_u8

def allowed_file(filename):
    """Check file extension against allowed upload formats."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/processor')
def processor():
    return render_template('processor.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Validate multipart upload and persist original image in uploads/.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 400
    
    h, w = image.shape[:2]
    
    return jsonify({
        'success': True,
        'filename': filename,
        'path': filepath,
        'dimensions': {'width': w, 'height': h}
    })

@app.route('/api/process', methods=['POST'])
def process():
    # Trigger segmentation pipeline and return preview URL plus metrics.
    data = request.json
    filepath = data.get('filepath')
    confidence = float(data.get('confidence', 0.3))
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_result.jpg")
        
        output_image, metrics = process_image(
            filepath,
            output_path=output_path,
            conf_threshold=confidence,
            return_metrics=True
        )
        
        # Verify output file was created
        if not os.path.exists(output_path):
            return jsonify({'error': 'Failed to create output file'}), 500
        
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Convert for PIL save compatibility.
        from PIL import Image
        pil_image = Image.fromarray(output_rgb)
        
        preview_filename = f"{name}_preview.jpg"
        preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
        pil_image.save(preview_path, 'JPEG')
        
        # Verify preview file was created
        if not os.path.exists(preview_path):
            return jsonify({'error': 'Failed to create preview file'}), 500
        
        return jsonify({
            'success': True,
            'output_path': output_path,
            'preview_url': f'/output/{preview_filename}?t={int(os.path.getmtime(preview_path))}',
            'output_filename': os.path.basename(output_path),
            'confidence': confidence,
            'metrics': {
                'total_objects_detected': metrics['total_objects_detected'],
                'objects_after_filter': metrics['objects_after_filter'],
                'avg_confidence': round(metrics['avg_confidence'], 4),
                'max_confidence': round(metrics['max_confidence'], 4),
                'min_confidence': round(metrics['min_confidence'], 4),
                'model_accuracy': round(metrics['model_accuracy'], 2),
                'total_mask_coverage': round(metrics['total_mask_coverage'], 2),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/heatmap', methods=['POST'])
def generate_heatmap():
    # Build and store a color-intensity heatmap overlay from the uploaded image.
    data = request.json
    filepath = data.get('filepath')
    alpha = float(data.get('alpha', 0.45))
    alpha = max(0.1, min(alpha, 0.9))

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_heatmap.jpg")

        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            return jsonify({'error': 'Failed to read image'}), 400

        heatmap_overlay, intensity_map = build_color_intensity_heatmap(image_bgr, alpha=alpha)
        cv2.imwrite(output_path, heatmap_overlay)

        preview_filename = f"{name}_heatmap_preview.jpg"
        preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
        cv2.imwrite(preview_path, heatmap_overlay)

        if not os.path.exists(output_path) or not os.path.exists(preview_path):
            return jsonify({'error': 'Failed to create heatmap file'}), 500

        hot_pixel_ratio = float((intensity_map >= 200).sum()) / float(intensity_map.size) * 100.0

        return jsonify({
            'success': True,
            'output_path': output_path,
            'preview_url': f'/output/{preview_filename}?t={int(os.path.getmtime(preview_path))}',
            'output_filename': os.path.basename(output_path),
            'heatmap_stats': {
                'mean_intensity': round(float(np.mean(intensity_map)), 2),
                'max_intensity': int(np.max(intensity_map)),
                'min_intensity': int(np.min(intensity_map)),
                'hot_pixel_ratio': round(hot_pixel_ratio, 2),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download(filename):
    # Send processed artifact as attachment to the browser.
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)

@app.route('/uploads/<filename>')
def get_upload(filename):
    # Serve uploaded source file for front-end preview.
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Upload file not found: {filepath}")
        return jsonify({'error': 'File not found'}), 404
    
    print(f"Serving upload: {filepath}")
    return send_file(filepath, mimetype='image/jpeg')

@app.route('/output/<filename>')
def get_output(filename):
    # Serve generated output file for front-end preview.
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Output file not found: {filepath}")
        return jsonify({'error': 'File not found'}), 404
    
    print(f"Serving output: {filepath}")
    return send_file(filepath, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
