# modules/dataset.py
# 1 Sobel , no pw using dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import re
from datetime import datetime, timedelta
from modules_1sobel.sobel import sobel_edges  # Import sobel_edges

print("1 sobel dataset used")


class TimeSeriesDataset(Dataset):

    def __init__(self, image_dir, transform=None, split='train', lp=0):  # lp를 생성자 파라미터로 추가
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        self.transform = transform
        self.split = split
        self.valid_sequences = self._find_valid_sequences()
        self.lp = lp  # lp를 인스턴스 변수로 저장

    def _find_valid_sequences(self):
        valid_sequences = []
        for i in range(len(self.image_files) - 9): 
            times = [self._extract_time(file) for file in self.image_files[i:i + 11]]  
            if all(times) and self._is_30min_interval(times):
                valid_sequences.append({
                    'index': i,
                    'start_time': times[0],
                    'end_time': times[-1]
                })
        return valid_sequences

    def _extract_time(self, file_path):  # 나의 경우
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{4})', file_path.name)
        if match:
            year, month, day, time = match.groups()
            return datetime.strptime(f"{year}{month}{day}{time}", "%Y%m%d%H%M")
        return None

    def _is_30min_interval(self, times):
        return all((times[i + 1] - times[i]) == timedelta(minutes=30) for i in range(len(times) - 1))

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        sequence_start = self.valid_sequences[idx]
        input_image = []

        # 5장의 이미지 로드 및 transform
        for i in range(5):
            # 이미지 차례로 불러오기
            img = Image.open(self.image_files[sequence_start['index'] + i]).convert("L")

            # 시퀀스 마지막 이미지를 부르기전, pw를 위한 이미지를 저장(lp변수를 받음)
            if i == 4:
                img_for_pw = Image.open(self.image_files[sequence_start['index'] + i - self.lp - 1]).convert("L") 

            if self.transform:
                img = self.transform(img)
                # 마지막 이미지
                if i == 4:
                    img_for_pw = self.transform(img_for_pw)

                    # new_power 1 with lag (Sobel Edge)
                    sobel_pw_ = sobel_edges(img) - sobel_edges(img_for_pw)  
                    sobel_pw = sobel_pw_ ** 2
                    
                    sobel_pw = (sobel_pw / (sobel_pw.max() + 1e-9)) * 2.0 - 1.0  
                    sobel_pw = sobel_pw.unsqueeze(0)  # Add a channel dimension
                    #print(f"image sobel data range:  [{sobel_pw.min()},   {sobel_pw.max()}]")
                    input_image.append(sobel_pw)
            #print(f"Image {i+1} data range: [{img.min()}, {img.max()}]")
            input_image.append(img)
            
            


        # 타겟 이미지 및 파일명 로드
        target_img_path = self.image_files[sequence_start['index'] + 5 + self.lp] 
        img_t = Image.open(target_img_path).convert("L")
        if self.transform:
            target = self.transform(img_t)

        filename = target_img_path.name

        return torch.cat(input_image, dim=0), target, filename