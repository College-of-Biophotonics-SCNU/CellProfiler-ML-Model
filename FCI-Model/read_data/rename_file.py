import os

def find_first_image_file(directory_path):
    # 获取指定目录下的所有文件名
    filenames = os.listdir(directory_path)

    # 过滤出以'image_'开头的文件名
    image_files = [filename for filename in filenames if filename.startswith('image_')]

    # 如果没有符合条件的文件，则返回None
    if not image_files:
        return None

    # 对文件名进行排序
    sorted_image_files = sorted(image_files)

    # 返回排序后的第一个文件名
    return sorted_image_files[0]


def rename_image_file(folder_path, newname):
    sub_folders = os.listdir(folder_path)
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            print("开始遍历实验组:  ", sub_folder)
            # 遍历不同实验组上的不同视野的图片
            sub_site_folders = os.listdir(sub_folder_path)
            for sub_site_folder in sub_site_folders:
                sub_site_folder_path = os.path.join(sub_folder_path, sub_site_folder)
                find_image = find_first_image_file(sub_site_folder_path)
                if os.path.isfile(os.path.join(sub_site_folder_path, newname)):
                    continue
                # 重新命名为为 newname
                os.rename(os.path.join(sub_site_folder_path, find_image), os.path.join(sub_site_folder_path, newname))


if __name__ == "__main__":
    folder_path = "D://data//20240708"
    rename_image_file(folder_path, "Hoechst.tif")