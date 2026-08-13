import os
import shutil

def flatten_image_folders(root_folder):
    count = 0

    for subdir, dirs, files in os.walk(root_folder):
        # 忽略根資料夾自己
        if subdir == root_folder:
            continue

        for file in files:
            # 只處理圖片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                src_path = os.path.join(subdir, file)

                # 建立新檔名：前綴為子資料夾名稱
                folder_name = os.path.basename(subdir)
                new_name = f"{folder_name}_{file}"
                dst_path = os.path.join(root_folder, new_name)

                # 如有同名檔案，自動編號避免衝突
                base, ext = os.path.splitext(new_name)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(root_folder, f"{base}_{counter}{ext}")
                    counter += 1

                # 搬移檔案
                shutil.move(src_path, dst_path)
                count += 1

    print(f"[INFO] Moved {count} images to {root_folder}")

    # 移除所有空資料夾
    for subdir, dirs, files in os.walk(root_folder, topdown=False):
        if subdir != root_folder and not os.listdir(subdir):
            os.rmdir(subdir)
            print(f"[INFO] Removed empty folder: {subdir}")

if __name__ == "__main__":
    # 修改為你的圖片資料夾路徑
    root_folder = r"Z:\Workspace\Jay\n2v_project\images"
    flatten_image_folders(root_folder)
