import cv2
import numpy as np
import os
from sklearn.metrics import classification_report

class BankNoteRecognitionSystem:
    def __init__(self):
        # RESTORED: nfeatures=0 allows maximum keypoint extraction
        self.sift = cv2.SIFT_create(nfeatures=0)
        
        FLANN_INDEX_KDTREE = 1
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=50)
        )
        
    def preprocess_method_a(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def preprocess_method_b(self, img):
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        return cv2.merge([h, s, v_clahe])

    def extract_local_features(self, gray_img, mask=None):
        keypoints, descriptors = self.sift.detectAndCompute(gray_img, mask)
        return keypoints, descriptors

    def extract_global_histogram(self, hsv_img, mask=None):
        hist = cv2.calcHist([hsv_img], [0, 1, 2], mask, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
    
    # ============================================================
    # STAGE 2: SEGMENTATION (DECOUPLED ALGORITHMS)
    # ============================================================
    def segment_for_strategy_a(self, img):
        """Method 1: Four-Point Perspective Transformation Warp (For SIFT)"""
        blurred = cv2.medianBlur(img, 15)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 120)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = sorted([c for c in contours if cv2.contourArea(c) > 5000], key=cv2.contourArea, reverse=True)

        if not valid:
            return img, np.ones(img.shape[:2], dtype=np.uint8) * 255

        best_quad = None
        for c in valid:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                if cv2.contourArea(approx) > 5000:
                    best_quad = approx
                    break

        if best_quad is None:
            x, y, w, h = cv2.boundingRect(valid[0])
            cropped = img[y:y+h, x:x+w]
            mask = np.ones(cropped.shape[:2], dtype=np.uint8) * 255
            return cropped, mask

        pts = best_quad.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   
        rect[2] = pts[np.argmax(s)]   
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  
        rect[3] = pts[np.argmax(diff)]  

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        mask = np.ones(warped.shape[:2], dtype=np.uint8) * 255
        return warped, mask

    def segment_for_strategy_b(self, img):
        """Method 2: Exact Contour Masking (For Histograms)"""
        blurred = cv2.medianBlur(img, 15) 
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 120)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = sorted([c for c in contours if cv2.contourArea(c) > 5000], key=cv2.contourArea, reverse=True)
        
        if not valid_contours:
            return img, None

        best_contour = None
        for c in valid_contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            is_valid_ratio = (1.0 < aspect_ratio < 3.0) or (0.3 < aspect_ratio < 1.0)
            
            if len(approx) >= 4 and len(approx) <= 6 and is_valid_ratio:
                best_contour = c
                break
                
        if best_contour is None:
            best_contour = valid_contours[0]

        x, y, w, h = cv2.boundingRect(best_contour)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        
        cropped_img = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        return cropped_img, cropped_mask

    def classify_strategy_a(self, query_kp, query_desc, reference_gallery):
        if query_desc is None or len(query_desc) < 4: 
            return "Unknown"
        
        best_label = "Unknown"
        best_confidence = 0.0

        for label, ref_list in reference_gallery.items():
            for ref_kp, ref_desc in ref_list:
                if ref_desc is None or len(ref_desc) < 4: 
                    continue
                
                try:
                    matches = self.flann.knnMatch(query_desc, ref_desc, k=2)
                except Exception:
                    continue
                
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2: 
                        m, n = m_n
                        if m.distance < 0.75 * n.distance: 
                            good_matches.append(m)
                
                if len(good_matches) < 10: 
                    continue
                
                src_pts = np.float32([ query_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                
                try:
                    M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
                except Exception:
                    continue
                
                if inlier_mask is None:
                    continue
                
                inliers = np.sum(inlier_mask)
                inlier_ratio = inliers / len(good_matches)

                confidence = inlier_ratio * np.sqrt(inliers)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label

        if best_confidence > 0.5:
            return best_label
        else:
            return "Unknown"

    def classify_strategy_b(self, query_hist, reference_gallery):
        best_match_label = "Unknown"
        max_score = -1.0 
        
        for label, ref_list in reference_gallery.items():
            for ref_hist in ref_list:
                score = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_INTERSECT)
                if score > max_score:
                    max_score = score
                    best_match_label = label
                    
        return best_match_label if max_score > 0.15 else "Unknown"

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    REF_DIR = "reference_templates"
    TEST_DIR = "test_images"
    
    system = BankNoteRecognitionSystem()
    
    gallery_a_sift = {} 
    gallery_b_hist = {}

    print("[INFO] Loading Multi-Template Reference Galleries...")
    
    if os.path.exists(REF_DIR):
        for denomination in os.listdir(REF_DIR):
            denom_path = os.path.join(REF_DIR, denomination)
            if not os.path.isdir(denom_path): continue
            
            gallery_a_sift[denomination] = []
            gallery_b_hist[denomination] = []
                
            for img_name in os.listdir(denom_path):
                img_path = os.path.join(denom_path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue

                # === PIPELINE A (SIFT) ===
                warped_a, mask_a = system.segment_for_strategy_a(img)
                warped_a = cv2.resize(warped_a, (600, 300))
                mask_a = cv2.resize(mask_a, (600, 300))
                
                prep_a = system.preprocess_method_a(warped_a)
                kp, descs = system.extract_local_features(prep_a, mask=mask_a)
                gallery_a_sift[denomination].append((kp, descs))

                # === PIPELINE B (Histogram) ===
                cropped_b, mask_b = system.segment_for_strategy_b(img)
                # Handle cases where segmentation fails
                if cropped_b is None:
                    cropped_b = cv2.resize(img, (600, 300))
                    mask_b = None
                else:
                    cropped_b = cv2.resize(cropped_b, (600, 300))
                    mask_b = cv2.resize(mask_b, (600, 300)) if mask_b is not None else None
                
                prep_b = system.preprocess_method_b(cropped_b)
                hist = system.extract_global_histogram(prep_b, mask=mask_b)
                gallery_b_hist[denomination].append(hist)
                
                print(f"[*] Loaded template: {img_name} for {denomination}")

    print("\n[INFO] Starting Testing Phase...")
    
    y_true = []
    y_pred_a = []
    y_pred_b = []
    
    if os.path.exists(TEST_DIR):
        for denomination in os.listdir(TEST_DIR):
            denom_path = os.path.join(TEST_DIR, denomination)
            if not os.path.isdir(denom_path): continue
                
            for img_name in os.listdir(denom_path):
                img_path = os.path.join(denom_path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue

                # === PIPELINE A (SIFT) ===
                warped_a, mask_a = system.segment_for_strategy_a(img)
                warped_a = cv2.resize(warped_a, (600, 300))
                mask_a = cv2.resize(mask_a, (600, 300))
                
                prep_a = system.preprocess_method_a(warped_a)
                kp, descs = system.extract_local_features(prep_a, mask=mask_a)
                pred_a = system.classify_strategy_a(kp, descs, gallery_a_sift)

                # === PIPELINE B (Histogram) ===
                cropped_b, mask_b = system.segment_for_strategy_b(img)
                if cropped_b is None:
                    cropped_b = cv2.resize(img, (600, 300))
                    mask_b = None
                else:
                    cropped_b = cv2.resize(cropped_b, (600, 300))
                    mask_b = cv2.resize(mask_b, (600, 300)) if mask_b is not None else None
                    
                prep_b = system.preprocess_method_b(cropped_b)
                hist = system.extract_global_histogram(prep_b, mask=mask_b)
                pred_b = system.classify_strategy_b(hist, gallery_b_hist)
                
                print(f"Test: {img_name} -> True: {denomination} | Pred A: {pred_a} | Pred B: {pred_b}")
                
                y_true.append(denomination)
                y_pred_a.append(pred_a)
                y_pred_b.append(pred_b)

        print("\n==========================================")
        print("          FINAL ACCURACY REPORT")
        print("==========================================")
        if len(y_true) > 0:
            total_tests = len(y_true)
            
            correct_a = sum(1 for t, p in zip(y_true, y_pred_a) if t == p)
            correct_b = sum(1 for t, p in zip(y_true, y_pred_b) if t == p)
            
            acc_a = (correct_a / total_tests) * 100
            acc_b = (correct_b / total_tests) * 100
            
            print(f"Total Images Tested : {total_tests}")
            print(f"Strategy A (SIFT)   : {acc_a:.2f}% ({correct_a}/{total_tests})")
            print(f"Strategy B (Hist)   : {acc_b:.2f}% ({correct_b}/{total_tests})")
            
            print("\n--- Strategy A Metrics ---")
            print(classification_report(y_true, y_pred_a, zero_division=0))
            
            print("--- Strategy B Metrics ---")
            print(classification_report(y_true, y_pred_b, zero_division=0))
        else:
            print("[WARNING] No test images were found to evaluate.")
        print("==========================================")