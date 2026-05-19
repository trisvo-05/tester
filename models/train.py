import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# VÌ CÁC FILE ĐÃ NẰM CẠNH NHAU, TA GỌI TRỰC TIẾP TÊN FILE
from dataset import MedSGJointDataset, collate_fn_filter_none
from triscatell_vlm import TriSliceVLM  # <-- Xóa chữ "models." ở đây

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {device}")

    # Tiền xử lý ảnh (Resize về 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Thư mục gốc chứa bộ dữ liệu (Nằm ở ổ D ngoài cùng)
    root_dir = r"D:\Data_AI\MedSG-Train"
    
    # Khởi tạo Dataset gom 8 Task (mỗi cụm rút ra 5 ảnh)
    try:
        dataset = MedSGJointDataset(root_dir=root_dir, num_slices=5, transform=transform)
        if len(dataset) == 0:
            print("⚠️ Không tìm thấy mẫu dữ liệu nào. Vui lòng kiểm tra lại cấu trúc JSON.")
            return
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_filter_none)
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo Dataset: {e}")
        return

    print("\n🔥 Bắt đầu chạy thử Dữ liệu (Mock Training Loop)...")
    valid_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue 
            
        cluster_tensor, questions, answers, bboxes = batch
        valid_batches += 1
        
        print(f"\n--- Batch {valid_batches} ---")
        print(f"Shape cụm ảnh: {cluster_tensor.shape}") 
        print(f"Câu hỏi: {questions[0]}")
        print(f"Tọa độ Bbox mục tiêu: {bboxes[0].tolist()}")
        
        # Dừng lại sau khi in thành công 3 batch
        if valid_batches == 3: 
            break
            
    print("\n✅ HOÀN TẤT! Pipeline dữ liệu chạy hoàn hảo.")

if __name__ == "__main__":
    main()