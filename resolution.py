import os
import torch
import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from modelsR import SRCNN
from utils import calc_psnr

def create_h5_from_image(image_path, h5_path):
    """Crea archivo HDF5 temporal para el procesamiento"""
    try:
        image = Image.open(image_path).convert('L')
        original = np.array(image).astype(np.float32) / 255.0

        downscale = image.resize((image.width // 2, image.height // 2), Image.BICUBIC)
        degraded = downscale.resize(image.size, Image.BICUBIC)
        degraded = np.array(degraded).astype(np.float32) / 255.0

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("hr", data=[original])
            hf.create_dataset("lr", data=[degraded])
            
        return True
    except Exception as e:
        print(f"Error creando HDF5: {str(e)}")
        return False

def run_super_resolution(model_path, h5_file, output_dir):
    """Ejecuta el modelo de superresolución y retorna el PSNR"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = SRCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        os.makedirs(output_dir, exist_ok=True)

        with h5py.File(h5_file, "r") as hf:
            lr = hf["lr"][0].astype(np.float32)
            hr = hf["hr"][0].astype(np.float32)

            input_tensor = torch.tensor(lr).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor).clamp(0.0, 1.0)

            ToPILImage()(torch.tensor(hr).unsqueeze(0)).save(os.path.join(output_dir, "original.png"))
            ToPILImage()(torch.tensor(lr).unsqueeze(0)).save(os.path.join(output_dir, "degradada.png"))
            ToPILImage()(output.squeeze().cpu()).save(os.path.join(output_dir, "superresolucion.png"))

            psnr = calc_psnr(output, torch.tensor(hr).unsqueeze(0).unsqueeze(0).to(device))
            print(f"PSNR (SRCNN vs Original): {psnr:.2f} dB")
            
        return True, psnr.item() 
    except Exception as e:
        print(f"Error en superresolución: {str(e)}")
        return False, 0.0

def process_image(image_path):
    print("[INFO] Procesando imagen...")
    temp_h5 = "temp_input.h5"
    output_dir = os.path.join("static", "resultados")
    
    if not create_h5_from_image(image_path, temp_h5):
        print("[ERROR] No se pudo crear el archivo HDF5")
        return None
    
    print("[INFO] HDF5 creado")

    success, psnr = run_super_resolution("SuperResolution.pth", temp_h5, output_dir)
    if not success:
        print("[ERROR] Falló la superresolución")
        return None
    
    print("[INFO] Superresolución exitosa")

    if os.path.exists(temp_h5):
        os.remove(temp_h5)
    
    return {
        'original': os.path.join("static", "resultados", "original.png").replace("\\", "/"),
        'degradada': os.path.join("static", "resultados", "degradada.png").replace("\\", "/"),
        'superresolucion': os.path.join("static", "resultados", "superresolucion.png").replace("\\", "/"),
        'psnr': round(psnr, 2) 
    }

def main():
    image_path = input("Ingresa la ruta de la imagen: ").strip()
    
    if not os.path.exists(image_path):
        print("La imagen no existe.")
        return
    
    result = process_image(image_path)
    if result:
        print(f"Procesamiento exitoso. Imágenes guardadas en: {result}")
        print(f"PSNR obtenido: {result['psnr']} dB")

if __name__ == "__main__":
    main()