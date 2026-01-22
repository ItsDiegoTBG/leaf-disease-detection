import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from ultralytics import YOLO
from scipy import ndimage

class ImageModelUI:
    def __init__(self, root):
        self.root = root
        root.title("Detector de enfermedad en plantas")
        root.geometry("900x650")
        root.resizable(True, True)

        self.current_display_image = None
        self.cv_image = None
        self.annotated_image = None
        self.segmentation_image = None
        self.showing_segmentation = False
        self.leaf_rois = []
        self.disease_data = []

        try:
            self.yolo_model = YOLO("yolo11x_leaf.pt")
            model_status = "YOLO model loaded successfully."
            model_color = "green"
        except Exception as e:
            self.yolo_model = None
            model_status = f"YOLO model load FAILED: {e}"
            model_color = "red"

        # Control bar
        control_frame = tk.Frame(root, padx=8, pady=8)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="Select Image",
                  command=self.on_select_image).pack(side=tk.LEFT, padx=6)

        tk.Button(control_frame, text="Open Camera",
                  command=self.on_open_camera).pack(side=tk.LEFT, padx=6)

        tk.Button(control_frame, text="Detect Leaves & Analyze",
                  command=self.on_detect_and_analyze).pack(side=tk.LEFT, padx=6)

        tk.Button(control_frame, text="Toggle Segmentation View",
                  command=self.toggle_segmentation).pack(side=tk.LEFT, padx=6)

        # Status label
        self.status_label = tk.Label(control_frame, text=model_status, fg=model_color)
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Main preview area
        preview_frame = tk.Frame(root, padx=8, pady=8)
        preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_frame, bg="#222", width=640, height=480)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Right panel with scrollbar
        right_frame = tk.Frame(preview_frame, width=260)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Label(right_frame, text="Detection Info",
                 font=("Arial", 11, "bold")).pack(anchor="nw")

        scroll_frame = tk.Frame(right_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(scroll_frame, width=34, height=24, wrap="word",
                                 yscrollcommand=scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)

        self.log("UI initialized.")
        self.log(model_status)

    def log(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert("end", text + "\n")
        self.info_text.see("end")
        self.info_text.config(state=tk.DISABLED)

    def on_select_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return

        self.cv_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.annotated_image = None
        self.segmentation_image = None
        self.showing_segmentation = False
        self.leaf_rois = []
        self.disease_data = []
        self.contour_debug_image = None

        pil_img = Image.fromarray(self.cv_image)
        self.show_image(pil_img)

        self.status_label.config(text=f"Loaded: {path}", fg="green")
        self.log(f"[Loaded Image] {path}")

    def on_open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.config(text="No camera found.", fg="red")
            return

        self.status_label.config(text="Press SPACE to capture, ESC to cancel.", fg="orange")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera - Press SPACE", frame)
            k = cv2.waitKey(1)

            if k == 27: 
                break
            elif k == 32: 
                self.cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.annotated_image = None
                self.segmentation_image = None
                self.showing_segmentation = False
                self.leaf_rois = []
                self.disease_data = []
                self.contour_debug_image = None

                self.show_image(Image.fromarray(self.cv_image))
                self.log("Captured from camera.")
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_leaves_with_yolo(self, img):
        if self.yolo_model is None:
            self.log("YOLO model not loaded!")
            return []
        
        results = self.yolo_model(img, conf=0.25)

        boxes = results[0].boxes
        detections = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())   
            if hasattr(box, 'cls'):
                class_id = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            })
            
            self.log(f"Leaf {i+1} detected - Confidence: {confidence:.2f}, Box: ({x1}, {y1}) -> ({x2}, {y2})")
        
        
        return detections

    def preprocess_roi_for_contour(self, roi_img):
        # Aqui usamos lo siguiente Ecualizacion de Hisstoriamas para intentar de mejorar el contraste de las hojas con aqui sin contraste
        # Luego Unsharp Masking para buscar realzar los colores
        # Un Filtro Bilateral para buscar retener esos bordes.

        lab = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        gaussian_blur = cv2.GaussianBlur(enhanced_rgb, (9, 9), 10.0)
        unsharp_mask = cv2.addWeighted(enhanced_rgb, 1.5, gaussian_blur, -0.5, 0)
        
        bilateral = cv2.bilateralFilter(unsharp_mask, 9, 75, 75)     
        return bilateral
    
    def extract_leaf_contour(self, roi_img):   
        roi_area = roi_img.shape[0] * roi_img.shape[1]
        h, w = roi_img.shape[:2]
        
        preprocessed = self.preprocess_roi_for_contour(roi_img)
        
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        center_margin_h = int(h * 0.2)
        center_margin_w = int(w * 0.2)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[center_margin_h:h-center_margin_h, center_margin_w:w-center_margin_w] = 255
        
        seed_pixels = np.count_nonzero(mask)
        region_l = l[mask > 0]
        region_a = a[mask > 0]
        region_b = b[mask > 0]
        
        l_mean, l_std = region_l.mean(), region_l.std()
        a_mean, a_std = region_a.mean(), region_a.std()
        b_mean, b_std = region_b.mean(), region_b.std()
        
        l_tolerance = 2.5
        a_tolerance = 2.5
        b_tolerance = 2.5
        
        l_lower = max(0, int(l_mean - l_tolerance * l_std))
        l_upper = min(255, int(l_mean + l_tolerance * l_std))
        a_lower = max(0, int(a_mean - a_tolerance * a_std))
        a_upper = min(255, int(a_mean + a_tolerance * a_std))
        b_lower = max(0, int(b_mean - b_tolerance * b_std))
        b_upper = min(255, int(b_mean + b_tolerance * b_std))
        l_mask = cv2.inRange(l, l_lower, l_upper)
        a_mask = cv2.inRange(a, a_lower, a_upper)
        b_mask = cv2.inRange(b, b_lower, b_upper)
        color_similar = cv2.bitwise_and(l_mask, a_mask)
        color_similar = cv2.bitwise_and(color_similar, b_mask)
        
        similar_pixels = np.count_nonzero(color_similar)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        color_similar = cv2.morphologyEx(color_similar, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_similar = cv2.morphologyEx(color_similar, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        morphed_pixels = np.count_nonzero(color_similar)
        strategy1_success = False
        
        if morphed_pixels > 0:
            contours, _ = cv2.findContours(color_similar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            if contours:
                center_point = (w // 2, h // 2)
                main_contour = None
                
                for contour in contours:
                    if cv2.pointPolygonTest(contour, center_point, False) >= 0:
                        main_contour = contour
                        break
                
                if main_contour is None:
                    main_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(main_contour)
                contour_percentage = (contour_area / roi_area) * 100
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
                if 15 < contour_percentage < 90 and solidity > 0.6:
                    strategy1_success = True
                    clean_leaf_mask = np.zeros_like(gray)
                    cv2.drawContours(clean_leaf_mask, [main_contour], -1, 255, -1)
                    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    clean_leaf_mask = cv2.morphologyEx(clean_leaf_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
        if not strategy1_success:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges1 = cv2.Canny(blurred, 30, 90)
            edges2 = cv2.Canny(blurred, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges_dilated = cv2.dilate(edges, kernel, iterations=3)
            edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=4)
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                center_point = (w // 2, h // 2)
                best_contour = None
                min_distance = float('inf')
                
                for contour in contours:
                    if cv2.pointPolygonTest(contour, center_point, False) >= 0:
                        best_contour = contour
                        break
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dist = np.sqrt((cx - center_point[0])**2 + (cy - center_point[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                            best_contour = contour
                
                if best_contour is None:
                    best_contour = max(contours, key=cv2.contourArea)
                
                contour_area = cv2.contourArea(best_contour)
                contour_percentage = (contour_area / roi_area) * 100
                clean_leaf_mask = np.zeros_like(gray)
                cv2.drawContours(clean_leaf_mask, [best_contour], -1, 255, -1)
                kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                clean_leaf_mask = cv2.morphologyEx(clean_leaf_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
            else:
                margin_h = int(h * 0.1)
                margin_w = int(w * 0.1)
                clean_leaf_mask = np.zeros_like(gray)
                clean_leaf_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
        
        return clean_leaf_mask
    
    def extract_disease_mask(self, roi_img, clean_leaf_mask):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
        lower_healthy = np.array([35, 50, 40])
        upper_healthy = np.array([85, 255, 255])
        healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
        disease_mask = cv2.bitwise_and(clean_leaf_mask, cv2.bitwise_not(healthy_mask))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        disease_mask = cv2.medianBlur(disease_mask, 5)

        leaf_pixels = np.count_nonzero(clean_leaf_mask)
        diseased_pixels = np.count_nonzero(disease_mask)
        
        if leaf_pixels > 0:
            disease_percentage = (diseased_pixels / leaf_pixels) * 100
        else:
            disease_percentage = 0.0
        
        self.log(f"  Leaf pixels: {leaf_pixels}, Diseased pixels: {diseased_pixels}")
        
        return disease_mask, disease_percentage

    def on_detect_and_analyze(self):
        if self.cv_image is None:
            self.log("Please load an image first!")
            return

        if self.yolo_model is None:
            self.log("YOLO model not available!")
            print("[ERROR] YOLO model not available")
            return

        self.leaf_rois = []
        self.disease_data = []

        self.log("\n=== Starting Detection ===")
        print("\n" + "=" * 60)
        print("STARTING LEAF DETECTION AND ANALYSIS")
        print("=" * 60)
        
        detections = self.detect_leaves_with_yolo(self.cv_image)
        
        if not detections:
            self.status_label.config(text="No leaves detected!", fg="red")
            print("\n[RESULT] No leaves detected!")
            return

        self.log(f"\nTotal leaves detected: {len(detections)}")
        print(f"\n[ANALYSIS] Processing {len(detections)} detected leaves...")

        self.annotated_image = self.cv_image.copy()
        self.segmentation_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
        self.contour_debug_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for detection in detections:
            leaf_id = detection['id']
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            roi_img = self.cv_image[y1:y2, x1:x2].copy()
            
            self.log(f"\n[Processing Leaf {leaf_id}]")
            leaf_mask = self.extract_leaf_contour(roi_img)
            disease_mask, disease_pct = self.extract_disease_mask(roi_img, leaf_mask)
            self.leaf_rois.append(detection['bbox'])
            self.disease_data.append({
                'id': leaf_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'disease_pct': disease_pct,
                'roi_size': (x2-x1, y2-y1)
            })
            
            color = colors[(leaf_id - 1) % len(colors)]
            cv2.rectangle(self.annotated_image, (x1, y1), (x2, y2), color, 3)
            
            label = f"Leaf {leaf_id}: {disease_pct:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(self.annotated_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            cv2.putText(self.annotated_image, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            seg_roi = np.zeros_like(roi_img)
            
            healthy_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(disease_mask))
            
            seg_roi[healthy_mask > 0] = [0, 255, 0]
            seg_roi[disease_mask > 0] = [255, 0, 0]
            
            self.segmentation_image[y1:y2, x1:x2] = seg_roi
            
            contour_debug = np.zeros_like(roi_img)
            contour_debug[leaf_mask > 0] = [255, 255, 255]
            self.contour_debug_image[y1:y2, x1:x2] = contour_debug

        self.showing_segmentation = False
        self.show_image(Image.fromarray(self.annotated_image))
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "===== DETECTION RESULTS =====\n\n")
        self.info_text.insert(tk.END, f"Total Leaves Detected: {len(detections)}\n")
        self.info_text.insert(tk.END, "=" * 35 + "\n\n")
        
        total_disease_pct = sum(d['disease_pct'] for d in self.disease_data) / len(self.disease_data)
        for data in self.disease_data:
            self.info_text.insert(tk.END, f"--- LEAF {data['id']} ---\n")
            self.info_text.insert(tk.END, f"Confidence: {data['confidence']:.2f}\n")
            self.info_text.insert(tk.END, f"Position: ({data['bbox'][0]}, {data['bbox'][1]})\n")
            self.info_text.insert(tk.END, f"Size: {data['roi_size'][0]} x {data['roi_size'][1]} px\n")
            self.info_text.insert(tk.END, f"Diseased Area: {data['disease_pct']:.2f}%\n")
            self.info_text.insert(tk.END, f"Healthy Area: {100-data['disease_pct']:.2f}%\n")
            
            if data['disease_pct'] < 30:
                status = "Healthy"
            elif data['disease_pct'] < 70:
                status = "Mild Disease"
            else:
                status = "Severe Disease"
            
            self.info_text.insert(tk.END, f"Status: {status}\n")
            self.info_text.insert(tk.END, "\n")
        
        self.info_text.insert(tk.END, "=" * 35 + "\n")
        self.info_text.insert(tk.END, f"Average Disease: {total_disease_pct:.2f}%\n")
        
        self.info_text.config(state=tk.DISABLED)
        
        if total_disease_pct < 30:
            overall_status = "All leaves healthy"
            color = "green"
        elif total_disease_pct < 70:
            overall_status = "Mild disease detected"
            color = "orange"
        else:
            overall_status = "Severe disease detected"
            color = "red"
            
        self.status_label.config(text=f"{len(detections)} leaves - {overall_status}", fg=color)

    def toggle_segmentation(self):
        if self.cv_image is None:
            return

        if self.segmentation_image is None:
            self.log("No segmentation available. Run detection first.")
            return

        self.showing_segmentation = not self.showing_segmentation

        if self.showing_segmentation:
            self.show_image(Image.fromarray(self.segmentation_image))
            self.status_label.config(text="Segmentation View (Green=Healthy, Red=Disease)", fg="purple")
        else:
            self.show_image(Image.fromarray(self.annotated_image))
            self.status_label.config(text=f"{len(self.leaf_rois)} leaves detected", fg="green")

    def show_image(self, pil_img):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 640
        if canvas_height <= 1:
            canvas_height = 480
        
        img_width, img_height = pil_img.size
        
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scale_factor = max(width_ratio, height_ratio)  

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        self.current_display_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.current_display_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageModelUI(root)
    root.mainloop()