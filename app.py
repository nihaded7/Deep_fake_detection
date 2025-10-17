from flask import Flask, render_template, request, redirect, flash, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration des répertoires de téléchargement et des extensions autorisées
app.config['UPLOAD_FOLDER'] = 'video_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'fakedeep'  # Clé secrète pour les sessions

# Chargement des modèles pré-entraînés
model = load_model('DPFAKE1_model.h5')
model2 = load_model('RESNET50_model.h5')

# Assurez-vous que le dossier de téléchargement existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Fonction pour vérifier si le fichier a une extension autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fonction pour extraire les images d'une vidéo
def extract_video_frames(video_path, image_size=(224, 224), num_frames=10):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        return None

    # Si le nombre de frames dans la vidéo est insuffisant, on en crée un nombre suffisant aléatoirement
    frame_indices = (
        np.linspace(0, frame_count - 1, num_frames, dtype=int) 
        if frame_count >= num_frames 
        else np.concatenate([np.arange(frame_count), np.random.choice(np.arange(frame_count), num_frames - frame_count)]),
    )

    frames = []
    for frame_idx in frame_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_capture.read()
        if not ret:
            frames.append(np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversion BGR -> RGB
        frame = cv2.resize(frame, image_size)  # Redimensionner l'image
        frames.append(frame)

    video_capture.release()
    frames = np.array(frames) / 255.0  # Normalisation des images
    return np.array(frames)

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la page de détection Deepfake
@app.route('/deepfake')
def deepfake():
    return render_template('deepfake.html')

# Route pour tester le modèle DPFAKE1
@app.route('/test_model', methods=['GET', 'POST'])
def test_model():
    if request.method == 'POST':
        # Vérifier si un fichier vidéo est téléchargé
        if 'file' not in request.files:
            flash('Aucun fichier téléchargé!', 'error')
            return redirect(request.url)

        video = request.files['file']

        # Vérifier si le fichier a un nom
        if video.filename == '':
            flash('Aucun fichier sélectionné!', 'error')
            return redirect(request.url)

        # Vérifier si le fichier est valide
        if video and allowed_file(video.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)  # Sauvegarder la vidéo téléchargée

            # Extraire les frames de la vidéo
            frames = extract_video_frames(filepath, image_size=(224, 224), num_frames=10)
            if frames is None or len(frames) == 0:
                flash('Erreur lors du traitement de la vidéo. Veuillez réessayer.', 'error')
                return redirect(request.url)

            # Sauvegarder les images extraites des frames
            frame_images = []
            frame_folder = os.path.join('static', 'frames')
            os.makedirs(frame_folder, exist_ok=True)

            for idx, frame in enumerate(frames):
                frame_path = os.path.join(frame_folder, f'frame_{idx + 1}.png')
                cv2.imwrite(frame_path, (frame * 255).astype(np.uint8))  # Sauvegarde des images
                frame_images.append(f'frames/frame_{idx + 1}.png')  # Chemin relatif des images

            # Effectuer la prédiction avec le modèle
            predictions = model.predict(frames)
            average_predictions = np.mean(predictions, axis=0)
            final_class = np.argmax(average_predictions)
            labels = ["Fake", "Reel"]
            final_label = labels[final_class]  # Label de la prédiction finale

            # Afficher le résultat sur la page HTML
            return render_template(
                'test_model.html',
                result=final_label,
                probabilities=dict(zip(labels, map(round, average_predictions, [2, 2]))),
                frame_images=frame_images  # Passer les images extraites pour affichage
            )

        flash('Format de fichier invalide. Veuillez télécharger une vidéo valide.', 'error')
        return redirect(request.url)

    return render_template('test_model.html', result=None, probabilities=None, frame_images=None)

# Route pour tester le modèle RESNET50
@app.route('/test_model2', methods=['GET', 'POST'])
def test_model2():
    if request.method == 'POST':
        # Vérifier si un fichier vidéo est téléchargé
        if 'file' not in request.files:
            flash('Aucun fichier téléchargé!', 'error')
            return redirect(request.url)

        video = request.files['file']

        if video.filename == '':
            flash('Aucun fichier sélectionné!', 'error')
            return redirect(request.url)

        if video and allowed_file(video.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)

            # Extraire les frames de la vidéo
            frames = extract_video_frames(filepath, image_size=(224, 224), num_frames=10)
            if frames is None or len(frames) == 0:
                flash('Erreur lors du traitement de la vidéo. Veuillez réessayer.', 'error')
                return redirect(request.url)

            # Sauvegarder les images extraites des frames
            frame_images = []
            frame_folder = os.path.join('static', 'frames')
            os.makedirs(frame_folder, exist_ok=True)

            for idx, frame in enumerate(frames):
                frame_path = os.path.join(frame_folder, f'frame_{idx + 1}.png')
                cv2.imwrite(frame_path, (frame * 255).astype(np.uint8))
                frame_images.append(f'frames/frame_{idx + 1}.png')  # Ajouter les chemins des images

            # Effectuer la prédiction avec le modèle
            predictions = model2.predict(frames)
            average_predictions = np.mean(predictions, axis=0)
            final_class = np.argmax(average_predictions)
            labels = ["Fake", "Reel"]
            final_label = labels[final_class]

            # Afficher le résultat sur la page HTML
            return render_template(
                'test_model2.html',
                result=final_label,
                probabilities=dict(zip(labels, map(round, average_predictions, [2, 2]))),
                frame_images=frame_images 
            )

        flash('Format de fichier invalide. Veuillez télécharger une vidéo valide.', 'error')
        return redirect(request.url)

    return render_template('test_model2.html', result=None, probabilities=None, frame_images=None)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
