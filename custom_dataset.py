# custom_dataset.py
import numpy as np
import os
import cv2 as cv
import torch.utils.data as data
import torch

class LineSegmentDataset(data.Dataset):
    def __init__(self, data_file, img_dir, label_dir, param, is_train=True):
        super(LineSegmentDataset, self).__init__()
        self.param = param
        self.in_res = self.param.inres
        self.out_res = self.param.outres
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.is_train = is_train
        
        # Load paths from data file (train.txt or val.txt)
        with open(data_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        self.num_samples = len(self.image_paths)
        print(f'Loaded {self.num_samples} samples for {"training" if is_train else "validation"}')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get image path and load image
        img_path = self.image_paths[idx]
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # Load image
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")
        
        # Preprocess the image (same as in YorkDataset)
        inp = cv.resize(img, self.in_res)
        H, W, C = inp.shape
        hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)
        imgv0 = hsv[..., 2]
        imgv = cv.resize(imgv0, (0, 0), fx=1. / 4, fy=1. / 4, interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv.resize(imgv, (W, H), interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        inp = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        inp = (inp.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1)
        
        # Load labels (line segments)
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")
        line_segments = []
        
        # Read line segments from txt file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 4:  # x1, y1, x2, y2 format
                        x1, y1, x2, y2 = map(float, values[:4])
                        line_segments.append([x1, y1, x2, y2])
        
        # Convert line segments to numpy array
        line_segments = np.array(line_segments, dtype=np.float32) if line_segments else np.zeros((0, 4), dtype=np.float32)
        
        # Create ground truth heatmaps for training
        center_map, dis_map, line_map = self.create_ground_truth(line_segments, H, W)
        
        ret = {
            "input": inp,
            "center": center_map,
            "dis": dis_map,
            "line": line_map,
            "filename": img_name,
            "origin_img": img,
            "line_segments": line_segments
        }
        return ret
    
    def create_ground_truth(self, line_segments, H, W):
        """
        Create ground truth maps for training:
        - center_map: centers of line segments
        - dis_map: distance maps (4 channels for distance to endpoints)
        - line_map: line segment mask
        """
        # Scale to output resolution
        h_ratio = self.out_res[1] / H
        w_ratio = self.out_res[0] / W
        
        # Initialize output maps
        center_map = np.zeros((1, self.out_res[1], self.out_res[0]), dtype=np.float32)
        dis_map = np.zeros((4, self.out_res[1], self.out_res[0]), dtype=np.float32)
        line_map = np.zeros((1, self.out_res[1], self.out_res[0]), dtype=np.float32)
        
        for line in line_segments:
            x1, y1, x2, y2 = line
            # Scale coordinates to output resolution
            x1 = x1 * w_ratio
            y1 = y1 * h_ratio
            x2 = x2 * w_ratio
            y2 = y2 * h_ratio
            
            # Draw line on line map
            cv.line(line_map[0], (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=1)
            
            # Calculate center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Draw center point with gaussian
            sigma = 2.0
            radius = int(3 * sigma)
            x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
            gaussian = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
            
            cx_int, cy_int = int(cx), int(cy)
            h, w = gaussian.shape
            
            # Apply gaussian to center map
            x_min = max(0, cx_int - radius)
            x_max = min(self.out_res[0], cx_int + radius + 1)
            y_min = max(0, cy_int - radius)
            y_max = min(self.out_res[1], cy_int + radius + 1)
            
            g_x_min = max(0, -cx_int + radius)
            g_x_max = min(w, self.out_res[0] - cx_int + radius)
            g_y_min = max(0, -cy_int + radius)
            g_y_max = min(h, self.out_res[1] - cy_int + radius)
            
            center_map[0, y_min:y_max, x_min:x_max] = np.maximum(
                center_map[0, y_min:y_max, x_min:x_max],
                gaussian[g_y_min:g_y_max, g_x_min:g_x_max]
            )
            
            # Create distance map for endpoints
            for y in range(self.out_res[1]):
                for x in range(self.out_res[0]):
                    if line_map[0, y, x] > 0:
                        # Distance to start point (normalized)
                        dis_map[0, y, x] = (x - x1) / self.out_res[0]
                        dis_map[1, y, x] = (y - y1) / self.out_res[1]
                        # Distance to end point (normalized)
                        dis_map[2, y, x] = (x - x2) / self.out_res[0]
                        dis_map[3, y, x] = (y - y2) / self.out_res[1]
        
        return center_map, dis_map, line_map