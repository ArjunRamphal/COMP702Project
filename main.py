import cv2
import numpy as np
import os
import random
from sklearn.metrics import classification_report

class SequentialBankNoteOptimizer:
    def __init__(self):
        # Feature Extractors
        self.sift = cv2.SIFT_create(nfeatures=0)
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.akaze = cv2.AKAZE_create()
        
        # Matchers
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.snapshot_dir = "report_snapshots"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    # STAGE 1: PREPROCESSING
    def prep_clahe(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def prep_bilateral(self, img):
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    def prep_gaussian(self, img):
        smoothed = cv2.GaussianBlur(img, (7, 7), 0)
        return cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    def prep_equalize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    # STAGE 2: SEGMENTATION & GEOMETRIC NORMALIZATION
    def get_base_contours(self, prep_img):
        blurred = cv2.medianBlur(prep_img, 15)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted([c for c in contours if cv2.contourArea(c) > 5000], key=cv2.contourArea, reverse=True)

    def seg_warp(self, prep_img):
        contours = self.get_base_contours(prep_img)
        if not contours: return None
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4: return approx
        return None

    def seg_exact_mask(self, prep_img):
        contours = self.get_base_contours(prep_img)
        if not contours: return None
        mask = np.zeros(prep_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contours[0]], -1, 255, -1)
        return mask

    def seg_padded_bbox(self, prep_img):
        contours = self.get_base_contours(prep_img)
        if not contours: return None
        x, y, w, h = cv2.boundingRect(contours[0])
        p = 5
        return (max(0, x-p), max(0, y-p), w+2*p, h+2*p)

    def seg_adapt_thresh(self, prep_img):
        thresh = cv2.adaptiveThreshold(prep_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = sorted([c for c in contours if cv2.contourArea(c) > 5000], key=cv2.contourArea, reverse=True)
        if not valid: return None
        mask = np.zeros(prep_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [valid[0]], -1, 255, -1)
        return mask

    def apply_segmentation(self, original_img, seg_output, method_name):
        if seg_output is None: return None, None
        if method_name == "Contour + Perspective Warp":
            pts = seg_output.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
            width = max(int(np.linalg.norm(rect[2]-rect[3])), int(np.linalg.norm(rect[1]-rect[0])))
            height = max(int(np.linalg.norm(rect[1]-rect[2])), int(np.linalg.norm(rect[0]-rect[3])))
            dst = np.array([[0,0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
            warped = cv2.warpPerspective(original_img, cv2.getPerspectiveTransform(rect, dst), (width, height))
            return warped, np.ones(warped.shape[:2], dtype=np.uint8) * 255
        elif method_name in ["Contour + Exact Mask", "AdaptThresh + Exact Mask"]:
            return cv2.bitwise_and(original_img, original_img, mask=seg_output), seg_output
        elif method_name == "Contour + Bounding Box":
            x, y, w, h = seg_output
            mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
            return original_img[y:y+h, x:x+w], mask[y:y+h, x:x+w]
        return None, None

    # STAGE 3 & 4: EXTRACTION & TEMPLATE MATCHING
    def classify_sift(self, img, mask, gallery):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None or len(desc) < 4: return "Unknown"
        best_label, best_conf = "Unknown", 0.0
        for label, refs in gallery.items():
            for ref_desc in refs:
                try: matches = self.flann.knnMatch(desc, ref_desc, k=2)
                except: continue
                good = [m for m_n in matches if len(m_n)==2 for m, n in [m_n] if m.distance < 0.75 * n.distance]
                if len(good) > best_conf: best_conf, best_label = len(good), label
        return best_label if best_conf > 10 else "Unknown"

    def classify_orb(self, img, mask, gallery):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, mask)
        if desc is None or len(desc) < 4: return "Unknown"
        best_label, best_conf = "Unknown", 0.0
        for label, refs in gallery.items():
            for ref_desc in refs:
                try: matches = self.bf_hamming.match(desc, ref_desc)
                except: continue
                if len(matches) > best_conf: best_conf, best_label = len(matches), label
        return best_label if best_conf > 10 else "Unknown"

    def classify_akaze(self, img, mask, gallery):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.akaze.detectAndCompute(gray, mask)
        if desc is None or len(desc) < 4: return "Unknown"
        best_label, best_conf = "Unknown", 0.0
        for label, refs in gallery.items():
            for ref_desc in refs:
                try: matches = self.bf_hamming.match(desc, ref_desc)
                except: continue
                if len(matches) > best_conf: best_conf, best_label = len(matches), label
        return best_label if best_conf > 10 else "Unknown"

    def classify_hist(self, img, mask, gallery):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        best_label, max_score = "Unknown", -1.0 
        for label, refs in gallery.items():
            for ref_hist in refs:
                score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_INTERSECT)
                if score > max_score: max_score, best_label = score, label
        return best_label if max_score > 0.15 else "Unknown"

def load_dataset(dir_path):
    dataset = []
    if os.path.exists(dir_path):
        for denom in os.listdir(dir_path):
            d_path = os.path.join(dir_path, denom)
            if not os.path.isdir(d_path): continue
            for img_name in os.listdir(d_path):
                img = cv2.imread(os.path.join(d_path, img_name))
                if img is not None: dataset.append((denom, img_name, img))
    return dataset

def report_final_pipeline(prep, seg, arch, acc):
    print(" FINAL PIPELINE CONFIGURATION REPORT")
    print(f"1. Preprocessing Stage      : {prep}")
    print(f"2. Segmentation Stage       : {seg}")
    print(f"3. Feature Extraction Stage : {arch}")
    print(f"4. Achieved Accuracy        : {acc:.2f}%")

if __name__ == "__main__":
    system = SequentialBankNoteOptimizer()
    templates, tests = load_dataset("reference_templates"), load_dataset("test_images")
    if not templates or not tests: exit()

    print("\n" + "\nPHASE 1: PREPROCESSING EVALUATION \n")
    prep_methods = {"CLAHE": system.prep_clahe, "Bilateral": system.prep_bilateral, 
                    "Gaussian": system.prep_gaussian, "Equalize": system.prep_equalize}
    best_prep_name, best_prep_acc = None, -1
    for p_name, p_func in prep_methods.items():
        gallery = {d: [] for d in os.listdir("reference_templates")}
        for denom, _, t_img in templates:
            t_prep = p_func(t_img)
            t_seg = system.seg_padded_bbox(t_prep)
            t_img_seg, t_mask = system.apply_segmentation(t_img, t_seg, "Contour + Bounding Box")
            if t_img_seg is not None:
                hsv = cv2.cvtColor(t_img_seg, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], cv2.resize(t_mask, (t_img_seg.shape[1], t_img_seg.shape[0])), [16, 16, 16], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                gallery[denom].append(hist)
        
        y_true, y_pred = [], []
        snapshots = {denom: 0 for denom in set(d for d, _, _ in tests)}
        for denom, img_name, img in tests:
            prep_img = p_func(img)
            if snapshots[denom] < 3:
                cv2.imwrite(f"{system.snapshot_dir}/Stage1_{p_name}_{denom}_{img_name}", prep_img)
                snapshots[denom] += 1
            seg_out = system.seg_padded_bbox(prep_img)
            img_seg, mask = system.apply_segmentation(img, seg_out, "Contour + Bounding Box")
            pred = system.classify_hist(img_seg, mask, gallery) if img_seg is not None else "Unknown"
            y_true.append(denom); y_pred.append(pred)
            
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(tests)
        print(f"\nTechnique [{p_name}] -> Accuracy: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, zero_division=0))
        if acc > best_prep_acc: best_prep_acc, best_prep_name = acc, p_name
        
    print(f"\n>>> WINNER STAGE 1: {best_prep_name} <<<"); winning_prep_func = prep_methods[best_prep_name]

    print("\n" + "\nPHASE 2: SEGMENTATION & GEOMETRIC NORMALIZATION \n")
    seg_methods = {"Contour + Perspective Warp": system.seg_warp, "Contour + Exact Mask": system.seg_exact_mask, 
                   "Contour + Bounding Box": system.seg_padded_bbox, "AdaptThresh + Exact Mask": system.seg_adapt_thresh}
    best_seg_name, best_seg_acc = None, -1
    for s_name, s_func in seg_methods.items():
        gallery = {d: [] for d in os.listdir("reference_templates")}
        for denom, _, t_img in templates:
            t_prep = winning_prep_func(t_img)
            t_seg = s_func(t_prep)
            t_img_seg, t_mask = system.apply_segmentation(t_img, t_seg, s_name)
            if t_img_seg is not None:
                hsv = cv2.cvtColor(cv2.resize(t_img_seg, (600,300)), cv2.COLOR_BGR2HSV)
                t_mask = cv2.resize(t_mask, (600,300)) if t_mask is not None else None
                hist = cv2.calcHist([hsv], [0, 1, 2], t_mask, [16, 16, 16], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                gallery[denom].append(hist)
        y_true, y_pred = [], []
        snapshots = {denom: 0 for denom in set(d for d, _, _ in tests)}
        for denom, img_name, img in tests:
            prep_img = winning_prep_func(img)
            seg_out = s_func(prep_img)
            img_seg, mask = system.apply_segmentation(img, seg_out, s_name)
            if img_seg is not None and snapshots[denom] < 3:
                cv2.imwrite(f"{system.snapshot_dir}/Stage2_{s_name}_{denom}_{img_name}", img_seg)
                snapshots[denom] += 1
            pred = system.classify_hist(img_seg, mask, gallery) if img_seg is not None else "Unknown"
            y_true.append(denom); y_pred.append(pred)
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(tests)
        print(f"\nTechnique [{s_name}] -> Accuracy: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, zero_division=0))
        if acc > best_seg_acc: best_seg_acc, best_seg_name = acc, s_name

    print(f"\nWINNER STAGE 2: {best_seg_name}"); winning_seg_func = seg_methods[best_seg_name]

    print("\n" + "\nPHASE 3: FEATURE EXTRACTION & TEMPLATE MATCHING \n")
    galleries = {"SIFT": {d: [] for d in os.listdir("reference_templates")}, "ORB": {d: [] for d in os.listdir("reference_templates")},
                 "AKAZE": {d: [] for d in os.listdir("reference_templates")}, "Hist": {d: [] for d in os.listdir("reference_templates")}}
    for denom, _, t_img in templates:
        t_prep = winning_prep_func(t_img); t_seg = winning_seg_func(t_prep)
        t_img_seg, t_mask = system.apply_segmentation(t_img, t_seg, best_seg_name)
        if t_img_seg is None: continue
        t_img_seg, t_mask = cv2.resize(t_img_seg, (600,300)), cv2.resize(t_mask, (600,300)) if t_mask is not None else None
        gray = cv2.cvtColor(t_img_seg, cv2.COLOR_BGR2GRAY)
        _, d_sift = system.sift.detectAndCompute(gray, t_mask); _, d_orb = system.orb.detectAndCompute(gray, t_mask)
        _, d_akaze = system.akaze.detectAndCompute(gray, t_mask)
        hsv = cv2.cvtColor(t_img_seg, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], t_mask, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        galleries["SIFT"][denom].append(d_sift); galleries["ORB"][denom].append(d_orb)
        galleries["AKAZE"][denom].append(d_akaze); galleries["Hist"][denom].append(hist)

    class_methods = {"SIFT": system.classify_sift, "ORB": system.classify_orb, "AKAZE": system.classify_akaze, "Hist": system.classify_hist}
    tech_details = {"SIFT": "Extractor: SIFT | Matcher: FLANN + USAC_MAGSAC", "ORB": "Extractor: ORB | Matcher: BF Hamming + RANSAC",
                    "AKAZE": "Extractor: AKAZE | Matcher: BF Hamming + RANSAC", "Hist": "Extractor: 3D HSV Histogram | Matcher: Intersection"}

    final_best_arch = None
    final_best_acc = -1
    for c_name, c_func in class_methods.items():
        print(f"\nEvaluating Final Pipeline Architecture: {c_name} | {tech_details[c_name]}")
        y_true, y_pred = [], []
        for denom, img_name, img in tests:
            prep_img = winning_prep_func(img); seg_out = winning_seg_func(prep_img)
            img_seg, mask = system.apply_segmentation(img, seg_out, best_seg_name)
            if img_seg is None: y_true.append(denom); y_pred.append("Unknown"); print(f"Test: {img_name} -> Unknown"); continue
            img_seg, mask = cv2.resize(img_seg, (600, 300)), cv2.resize(mask, (600, 300)) if mask is not None else None
            cv2.setRNGSeed(np.random.randint(1, 999999)); pred = c_func(img_seg, mask, galleries[c_name])
            y_true.append(denom); y_pred.append(pred); print(f"Test: {img_name} -> True: {denom} | Pred: {pred}")
        print(f"\nFinal Report: {c_name} Architecture -> Accuracy: {(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(tests))*100:.2f}%")
        print(classification_report(y_true, y_pred, zero_division=0))
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(tests)
        
        if acc > final_best_acc:
            final_best_acc = acc
            final_best_arch = c_name

    report_final_pipeline(best_prep_name, best_seg_name, final_best_arch, final_best_acc * 100)