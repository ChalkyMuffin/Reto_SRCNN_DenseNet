from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from resolution import process_image
from predict import predict_image
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pgm'}
app.config['STATIC_IMAGES'] = ['training_graph.png', 'matriz_de_confusion.png', 'training_densenet_graph.png']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/templates/<path:filename>')
def templates_static(filename):
    return send_from_directory('templates', filename)

@app.route('/static/images/<path:filename>')
def static_images(filename):
    return send_from_directory('static/images', filename)

@app.route('/')
def home():
    image_urls = {}
    for img in app.config['STATIC_IMAGES']:
        image_urls[img.split('.')[0]] = url_for('static_images', filename=img)
    
    return render_template('index.html', **image_urls)

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)

        try:
            print("[INFO] Imagen subida en:", upload_path)
            images = process_image(upload_path)

            if not images:
                print("[ERROR] No se generaron imágenes.")
                raise Exception("Error en superresolución")

            original_rel = os.path.relpath(images['original'], os.path.join(BASE_DIR, 'static')).replace("\\", "/")
            degradada_rel = os.path.relpath(images['degradada'], os.path.join(BASE_DIR, 'static')).replace("\\", "/")
            superresolucion_rel = os.path.relpath(images['superresolucion'], os.path.join(BASE_DIR, 'static')).replace("\\", "/")

            sr_image_path = os.path.join(BASE_DIR, 'static', superresolucion_rel)
            clasificacion = predict_image(sr_image_path)

            return render_template('resultado.html',
                                   original=original_rel,
                                   degradada=degradada_rel,
                                   superresolucion=superresolucion_rel,
                                   clasificacion=clasificacion,
                                   psnr=images['psnr'])  

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return f'Error al procesar la imagen: {str(e)}'

    return 'Formato de archivo no válido'

def copy_static_images():
    """Copia todas las imágenes estáticas desde templates a static/images"""
    images_dir = os.path.join(BASE_DIR, 'static', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    for img in app.config['STATIC_IMAGES']:
        src = os.path.join(BASE_DIR, 'templates', img)
        dest = os.path.join(images_dir, img)
        
        if os.path.exists(src):
            shutil.copy(src, dest)
            print(f"[INFO] Copiada imagen estática: {img}")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "resultados"), exist_ok=True)
    
    copy_static_images()
    
    app.run(debug=True)