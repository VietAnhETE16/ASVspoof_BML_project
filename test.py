import torch
import numpy as np
import librosa
import yaml
import os
from model import RawNet 

# --- CẤU HÌNH ---

# 1. Đường dẫn đến file model .pth 
MODEL_PATH = "models/model_LA_weighted_CCE_100_32_0.0001/epoch_15.pth"

# 2. Đường dẫn đến file cấu hình YAML 
YAML_CONFIG = 'model_config_RawNet.yaml'

# 3. Đường dẫn file âm thanh cần kiểm tra
AUDIO_FILE = 'test_audio.mp3' 

# 4. Cấu hình thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, config_path):
    # Bước 1: Đọc cấu hình từ file YAML (để khớp size với model đã train)
    if not os.path.exists(config_path):
        print(f"Lỗi: Không tìm thấy file config tại {config_path}")
        print("Hãy đảm bảo file 'model_config_RawNet.yaml' nằm cùng thư mục với test.py")
        exit()
        
    with open(config_path, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)
        # Lấy tham số model từ key 'model' trong file yaml
        d_args = parser1['model'] 
    
    print(f"🔹 Cấu hình model loaded: first_conv={d_args['first_conv']}")

    # Bước 2: Khởi tạo model với cấu hình vừa đọc
    model = RawNet(d_args, device).to(device)
    
    # Bước 3: Load trọng số
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file model: {model_path}")
        exit()

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã load model thành công: {model_path}")
    except Exception as e:
        print(f"Lỗi load trực tiếp: {e}")
        # Thử load theo kiểu checkpoint nếu file pth chứa cả optimizer
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
             print("Đã load model từ checkpoint thành công.")
        else:
            print("Không thể load model (Vẫn bị lệch size hoặc sai file).")
            exit()
            
    model.eval() 
    return model

def process_audio(file_path):
    # Cắt hoặc đệm file âm thanh cho đủ độ dài chuẩn
    cut = 64600
    try:
        X, fs = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Lỗi thư viện Librosa không đọc được file: {e}")
        return None

    X_pad = np.zeros(cut)
    if X.shape[0] < cut:
        X_pad[:X.shape[0]] = X
    else:
        X_pad = X[:cut]
    
    # Chuyển thành Tensor
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)
    return x_inp

def predict(model, audio_path):
    tensor_audio = process_audio(audio_path)
    if tensor_audio is None:
        return

    with torch.no_grad():
        output = model(tensor_audio)
        probs = torch.nn.functional.softmax(output, dim=1)
        spoof_score = probs[0][0].item() * 100
        bonafide_score = probs[0][1].item() * 100
        
        print(f"\n--- KẾT QUẢ KIỂM TRA: {audio_path} ---")
        print(f"Giả mạo (Spoof): {spoof_score:.2f}%")
        print(f"Thật (Bonafide): {bonafide_score:.2f}%")
        
        if bonafide_score > spoof_score:
            print("=> KẾT LUẬN: ÂM THANH THẬT")
        else:
            print("=> KẾT LUẬN: ÂM THANH GIẢ MẠO")

if __name__ == "__main__":
    # Load model với file config
    model = load_model(MODEL_PATH, YAML_CONFIG)
    
    # Chạy thử
    if os.path.exists(AUDIO_FILE):
        predict(model, AUDIO_FILE)
    else:
        print(f"Chưa có file âm thanh mẫu: {AUDIO_FILE}")
        print("Hãy đổi biến AUDIO_FILE trong code thành đường dẫn file bạn muốn test.")