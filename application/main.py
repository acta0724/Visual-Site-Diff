#!/usr/bin/env python3
"""
画像差分検出システム - コマンドライン版
使用方法:
python image_diff_cli.py image1.jpg image2.jpg [オプション]

基本的な使用例:
python image_diff_cli.py before.jpg after.jpg

差分感度の調整:
python image_diff_cli.py img1.jpg img2.jpg --diff-threshold 20 --min-area 50

高精度モード:
python image_diff_cli.py img1.jpg img2.jpg --method SIFT --diff-threshold 15

結果保存:
python image_diff_cli.py img1.jpg img2.jpg --output results/ --verbose
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json
from typing import Tuple, List, Optional

class ImageDifferenceDetector:
    def __init__(self, feature_method='SIFT', match_threshold=0.75, 
                 diff_threshold=30, min_area=100, morph_kernel_size=5):
        """
        画像差分検出システムの初期化
        
        Args:
            feature_method: 特徴量抽出手法 ('ORB', 'SIFT', 'AKAZE')
            match_threshold: マッチング閾値
            diff_threshold: 差分画像の二値化閾値 (小さいほど敏感)
            min_area: 差分として認識する最小面積 (大きいほどノイズを除去)
            morph_kernel_size: モルフォロジー処理のカーネルサイズ
        """
        self.feature_method = feature_method
        self.match_threshold = match_threshold
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.morph_kernel_size = morph_kernel_size
        
        if feature_method == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif feature_method == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher()
        elif feature_method == 'AKAZE':
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError("Unsupported feature method")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        if desc1 is None or desc2 is None:
            return []
        
        if self.feature_method == 'SIFT':
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            matches = []
            for m_n in raw_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.match_threshold * n.distance:
                        matches.append(m)
        else:
            matches = self.matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if matches:
                good_matches = [m for m in matches if m.distance < 50]
                matches = good_matches[:100]
        
        return matches
    
    def estimate_transformation(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        if len(matches) < 4:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0
        )
        
        return homography, mask
    
    def detect_differences(self, img1: np.ndarray, img2: np.ndarray, 
                          homography: np.ndarray = None) -> np.ndarray:
        if homography is not None:
            h, w = img2.shape
            aligned_img1 = cv2.warpPerspective(img1, homography, (w, h))
        else:
            aligned_img1 = img1
        
        # 差分計算
        diff = cv2.absdiff(aligned_img1, img2)
        
        # 設定可能な閾値で二値化
        _, binary_diff = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 設定可能なカーネルサイズでモルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morph_kernel_size, self.morph_kernel_size))
        cleaned_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
        cleaned_diff = cv2.morphologyEx(cleaned_diff, cv2.MORPH_OPEN, kernel)
        
        return cleaned_diff, aligned_img1
    
    def find_difference_contours(self, diff_img: np.ndarray) -> List:
        contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 設定可能な最小面積でフィルタリング
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]
        return filtered_contours
    
    def compare_images(self, img1_path: str, img2_path: str) -> dict:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None:
            raise ValueError(f"Could not load image: {img1_path}")
        if img2 is None:
            raise ValueError(f"Could not load image: {img2_path}")
        
        proc_img1 = self.preprocess_image(img1)
        proc_img2 = self.preprocess_image(img2)
        
        kp1, desc1 = self.extract_features(proc_img1)
        kp2, desc2 = self.extract_features(proc_img2)
        
        matches = self.match_features(desc1, desc2)
        
        result = {
            'image1_path': img1_path,
            'image2_path': img2_path,
            'num_features_1': len(kp1),
            'num_features_2': len(kp2),
            'num_matches': len(matches),
            'similarity_score': len(matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        }
        
        if len(matches) >= 4:
            transformation_result = self.estimate_transformation(kp1, kp2, matches)
            if transformation_result is not None:
                homography, mask = transformation_result
                result['homography'] = homography
                result['inlier_matches'] = int(np.sum(mask))
                
                diff_img, aligned_img1 = self.detect_differences(proc_img1, proc_img2, homography)
                contours = self.find_difference_contours(diff_img)
                
                result['difference_image'] = diff_img
                result['aligned_image'] = aligned_img1
                result['difference_contours'] = contours
                result['num_differences'] = len(contours)
        
        return result

def save_results(result: dict, output_dir: str):
    """結果をファイルに保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON結果の保存（NumPy配列は除外）
    json_result = {}
    for key, value in result.items():
        if key not in ['difference_image', 'aligned_image', 'homography', 'difference_contours']:
            json_result[key] = value
        elif key == 'homography' and value is not None:
            json_result[key] = value.tolist()  # NumPy配列をリストに変換
    
    json_path = output_path / 'comparison_result.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"結果をJSONで保存: {json_path}")
    
    # 差分画像の保存
    if 'difference_image' in result and result['difference_image'] is not None:
        diff_path = output_path / 'difference_image.png'
        cv2.imwrite(str(diff_path), result['difference_image'])
        print(f"差分画像を保存: {diff_path}")
    
    # 位置合わせ後の画像の保存
    if 'aligned_image' in result and result['aligned_image'] is not None:
        aligned_path = output_path / 'aligned_image.png'
        cv2.imwrite(str(aligned_path), result['aligned_image'])
        print(f"位置合わせ画像を保存: {aligned_path}")
    
    # 差分をハイライトした画像の保存
    if 'difference_contours' in result and result['difference_contours']:
        img2 = cv2.imread(result['image2_path'])
        img_with_diff = img2.copy()
        cv2.drawContours(img_with_diff, result['difference_contours'], -1, (0, 255, 0), -1)
        highlight_path = output_path / 'differences_highlighted.png'
        cv2.imwrite(str(highlight_path), img_with_diff)
        print(f"差分ハイライト画像を保存: {highlight_path}")

def main():
    parser = argparse.ArgumentParser(description='画像差分検出システム')
    parser.add_argument('image1', help='基準画像のパス')
    parser.add_argument('image2', help='比較画像のパス')
    parser.add_argument('--method', choices=['ORB', 'SIFT', 'AKAZE'], 
                       default='ORB', help='特徴量抽出手法 (デフォルト: ORB)')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='マッチング閾値 (デフォルト: 0.75)')
    parser.add_argument('--diff-threshold', type=int, default=30,
                       help='差分検出の閾値 (0-255, 小さいほど敏感, デフォルト: 30)')
    parser.add_argument('--min-area', type=int, default=100,
                       help='差分として認識する最小面積 (大きいほどノイズ除去, デフォルト: 100)')
    parser.add_argument('--morph-kernel', type=int, default=5,
                       help='モルフォロジー処理のカーネルサイズ (デフォルト: 5)')
    parser.add_argument('--output', '-o', default=None,
                       help='結果保存フォルダ（指定しない場合は保存しない）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細情報を表示')
    
    args = parser.parse_args()
    
    # パラメータ範囲チェック
    if not (0 <= args.diff_threshold <= 255):
        print("エラー: --diff-threshold は 0-255 の範囲で指定してください")
        sys.exit(1)
    
    if args.min_area < 0:
        print("エラー: --min-area は 0 以上で指定してください")
        sys.exit(1)
    
    if args.morph_kernel < 3 or args.morph_kernel % 2 == 0:
        print("エラー: --morph-kernel は 3 以上の奇数で指定してください")
        sys.exit(1)
    
    # ファイル存在確認
    if not os.path.exists(args.image1):
        print(f"エラー: ファイルが見つかりません: {args.image1}")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"エラー: ファイルが見つかりません: {args.image2}")
        sys.exit(1)
    
    try:
        # 検出器を初期化（新しいパラメータを含む）
        detector = ImageDifferenceDetector(
            feature_method=args.method,
            match_threshold=args.threshold,
            diff_threshold=args.diff_threshold,
            min_area=args.min_area,
            morph_kernel_size=args.morph_kernel
        )
        
        if args.verbose:
            print(f"使用手法: {args.method}")
            print(f"マッチング閾値: {args.threshold}")
            print(f"差分検出閾値: {args.diff_threshold}")
            print(f"最小面積: {args.min_area}")
            print(f"モルフォロジーカーネル: {args.morph_kernel}")
            print(f"基準画像: {args.image1}")
            print(f"比較画像: {args.image2}")
            print("処理中...")
        
        # 画像を比較
        result = detector.compare_images(args.image1, args.image2)
        
        # 結果を表示
        print("\n=== 比較結果 ===")
        print(f"基準画像の特徴点数: {result['num_features_1']}")
        print(f"比較画像の特徴点数: {result['num_features_2']}")
        print(f"マッチした特徴点数: {result['num_matches']}")
        print(f"類似度スコア: {result['similarity_score']:.3f}")
        
        if args.verbose:
            print(f"差分検出パラメータ:")
            print(f"  - 差分閾値: {args.diff_threshold}")
            print(f"  - 最小面積: {args.min_area}")
            print(f"  - カーネルサイズ: {args.morph_kernel}")
        
        if 'inlier_matches' in result:
            print(f"有効マッチ数: {result['inlier_matches']}")
        
        if 'num_differences' in result:
            print(f"検出された差分領域数: {result['num_differences']}")
            
            if result['num_differences'] == 0:
                print("→ 有意な差分は検出されませんでした")
                if args.verbose:
                    print("  (より敏感にするには --diff-threshold を小さくするか --min-area を小さくしてください)")
            else:
                print(f"→ {result['num_differences']}個の差分領域を検出")
        else:
            print("→ 位置合わせができませんでした（類似点が不足）")
        
        # 結果を保存
        if args.output:
            save_results(result, args.output)
        
        print("\n処理完了！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()