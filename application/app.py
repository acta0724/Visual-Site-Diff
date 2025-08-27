#!/usr/bin/env python3
"""
画像差分検出システム - Flask Web版
"""

import os
import json
import shutil
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
from main import ImageDifferenceDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# アップロード用ディレクトリの設定
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_sessions():
    """古いセッションファイルを削除（最新5個を保持）"""
    try:
        # アップロードフォルダとリザルトフォルダの両方をクリーンアップ
        for folder_path in [Path(UPLOAD_FOLDER), Path(RESULTS_FOLDER)]:
            if not folder_path.exists():
                continue
                
            # セッションフォルダを作成時刻でソート（新しい順）
            session_dirs = []
            for session_dir in folder_path.iterdir():
                if session_dir.is_dir():
                    try:
                        # フォルダの作成時刻を取得
                        creation_time = session_dir.stat().st_ctime
                        session_dirs.append((creation_time, session_dir))
                    except OSError:
                        continue
            
            # 作成時刻で降順ソート（新しいものが先頭）
            session_dirs.sort(reverse=True, key=lambda x: x[0])
            
            # 6個目以降（インデックス5以降）を削除
            for _, old_dir in session_dirs[5:]:
                try:
                    shutil.rmtree(old_dir)
                    print(f"削除しました: {old_dir}")
                except OSError as e:
                    print(f"削除に失敗: {old_dir}, エラー: {e}")
                    
    except Exception as e:
        print(f"クリーンアップ中にエラー: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '2つの画像ファイルが必要です'}), 400
    
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': '画像ファイルを選択してください'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': '許可されていないファイル形式です'}), 400
    
    # 古いセッションをクリーンアップ（新しいセッション作成前に実行）
    cleanup_old_sessions()
    
    # ユニークなセッションIDを生成
    session_id = str(uuid.uuid4())
    session_folder = Path(UPLOAD_FOLDER) / session_id
    session_folder.mkdir(exist_ok=True)
    
    # ファイル保存
    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    
    filepath1 = session_folder / filename1
    filepath2 = session_folder / filename2
    
    file1.save(str(filepath1))
    file2.save(str(filepath2))
    
    # パラメータ取得
    method = request.form.get('method', 'ORB')
    threshold = float(request.form.get('threshold', 0.75))
    diff_threshold = int(request.form.get('diff_threshold', 30))
    min_area = int(request.form.get('min_area', 100))
    morph_kernel = int(request.form.get('morph_kernel', 5))
    
    try:
        # 画像差分検出実行
        detector = ImageDifferenceDetector(
            feature_method=method,
            match_threshold=threshold,
            diff_threshold=diff_threshold,
            min_area=min_area,
            morph_kernel_size=morph_kernel
        )
        
        result = detector.compare_images(str(filepath1), str(filepath2))
        
        # 結果保存用ディレクトリ作成
        result_folder = Path(RESULTS_FOLDER) / session_id
        result_folder.mkdir(exist_ok=True)
        
        # 結果をファイルに保存
        save_results_web(result, str(result_folder))
        
        # Web用レスポンス作成
        response = {
            'session_id': session_id,
            'success': True,
            'result': {
                'num_features_1': result.get('num_features_1', 0),
                'num_features_2': result.get('num_features_2', 0),
                'num_matches': result.get('num_matches', 0),
                'similarity_score': result.get('similarity_score', 0),
                'inlier_matches': result.get('inlier_matches', 0),
                'num_differences': result.get('num_differences', 0),
                'num_objects': result.get('num_objects', 0)
            },
            'images': {
                'original1': f'/uploads/{session_id}/{filename1}',
                'original2': f'/uploads/{session_id}/{filename2}',
                'difference': f'/results/{session_id}/difference_image.png',
                'aligned': f'/results/{session_id}/aligned_image.png',
                'highlighted': f'/results/{session_id}/differences_highlighted.png',
                'detection': f'/results/{session_id}/object_detection.png'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_results_web(result, output_dir):
    """Web用の結果保存"""
    output_path = Path(output_dir)
    
    # JSON結果の保存（NumPy配列は除外）
    json_result = {k: v for k, v in result.items() 
                   if k not in ['difference_image', 'aligned_image', 'difference_contours', 'bounding_boxes']}
    
    # ホモグラフィ行列をリストに変換
    if 'homography' in json_result and json_result['homography'] is not None:
        json_result['homography'] = json_result['homography'].tolist()
    
    json_path = output_path / 'comparison_result.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    # 差分画像の保存
    if 'difference_image' in result and result['difference_image'] is not None:
        import cv2
        diff_path = output_path / 'difference_image.png'
        cv2.imwrite(str(diff_path), result['difference_image'])
    
    # 位置合わせ後の画像の保存
    if 'aligned_image' in result and result['aligned_image'] is not None:
        aligned_path = output_path / 'aligned_image.png'
        cv2.imwrite(str(aligned_path), result['aligned_image'])
    
    # 差分をハイライトした画像の保存
    if 'difference_contours' in result and result['difference_contours']:
        import cv2
        if 'aligned_image' in result and result['aligned_image'] is not None:
            img_with_diff = result['aligned_image'].copy()
        else:
            # フォールバック
            img_with_diff = cv2.imread(result['image2_path'])
        
        cv2.drawContours(img_with_diff, result['difference_contours'], -1, (0, 0, 255), -1)
        highlight_path = output_path / 'differences_highlighted.png'
        cv2.imwrite(str(highlight_path), img_with_diff)
    
    # 物体検知画像の保存
    if 'bounding_boxes' in result and result['bounding_boxes']:
        import cv2
        if 'aligned_image' in result and result['aligned_image'] is not None:
            img_with_boxes = result['aligned_image'].copy()
        else:
            # フォールバック
            img_with_boxes = cv2.imread(result['image2_path'])
        
        # バウンディングボックスを描画
        for i, box in enumerate(result['bounding_boxes']):
            x, y, w, h = box
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_with_boxes, f'Object {i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        detection_path = output_path / 'object_detection.png'
        cv2.imwrite(str(detection_path), img_with_boxes)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/export_zip/<session_id>')
def export_zip(session_id):
    """セッションの全ファイルをZIPで出力"""
    import zipfile
    from io import BytesIO
    
    upload_folder = Path(UPLOAD_FOLDER) / session_id
    result_folder = Path(RESULTS_FOLDER) / session_id
    
    # ZIPファイルをメモリに作成
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # アップロード画像を追加
        if upload_folder.exists():
            for file_path in upload_folder.iterdir():
                if file_path.is_file():
                    zip_file.write(file_path, f'original_images/{file_path.name}')
        
        # 結果画像を追加
        if result_folder.exists():
            for file_path in result_folder.iterdir():
                if file_path.is_file():
                    zip_file.write(file_path, f'results/{file_path.name}')
    
    zip_buffer.seek(0)
    
    from flask import Response
    return Response(
        zip_buffer.getvalue(),
        headers={
            'Content-Type': 'application/zip',
            'Content-Disposition': f'attachment; filename=image_diff_{session_id[:8]}.zip'
        }
    )

@app.route('/cleanup')
def manual_cleanup():
    """手動でクリーンアップを実行"""
    try:
        cleanup_old_sessions()
        return jsonify({
            'success': True,
            'message': '古いセッションファイルをクリーンアップしました'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)