import os
import shutil
import re


def rename_and_copy_images(root_dir, output_dir, datatime, cell_line='None'):
    data_dir = root_dir + "/" + datatime
    exp_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir))]
    # 定义匹配规则
    exp_pattern = r'(\d+)_(\d+)'
    batch_pattern = r'(\d+)'
    for exp_folder in exp_folders:
        # 使用正则表达式进行匹配
        match = re.match(exp_pattern, exp_folder)
        hour_time = match.group(1)  # 实验的时间
        is_control = match.group(2)  # 实验是否为对照组
        batch_folders_name = data_dir + "/" + exp_folder
        for batch_folder in os.listdir(batch_folders_name):
            batch_match = re.match(batch_pattern, batch_folder)
            batch_name = batch_match.group(1)
            batch_folder_name = batch_folders_name + "/" + batch_folder
            batch_original_path = os.path.join(batch_folder_name, batch_folder_name)
            print("批次加载的文件目录：" + batch_original_path)
            file_pre_new_name = ('cl' + cell_line + '_exp' + datatime + '_h' + hour_time + '_b' + batch_name + '_ic'
                                 + is_control + '_c')
            shutil.copyfile(os.path.join(batch_original_path, "AA.tif"),
                            os.path.join(output_dir, file_pre_new_name + "AA.tif"))
            shutil.copyfile(os.path.join(batch_original_path, "DA.tif"),
                            os.path.join(output_dir, file_pre_new_name + "DA.tif"))
            shutil.copyfile(os.path.join(batch_original_path, "DD.tif"),
                            os.path.join(output_dir, file_pre_new_name + "DD.tif"))
            shutil.copyfile(os.path.join(batch_original_path + '/E-FRET results', 'ED.tif'),
                            os.path.join(output_dir, file_pre_new_name + "ED.tif"))
            shutil.copyfile(os.path.join(batch_original_path + '/E-FRET results', 'moban.tif'),
                            os.path.join(output_dir, file_pre_new_name + "MB.tif"))
            shutil.copyfile(os.path.join(batch_original_path + '/E-FRET results', 'Rc.tif'),
                            os.path.join(output_dir, file_pre_new_name + "RC.tif"))
            # 寻找第一张明场图像进行数据加载预处理
            # 定义匹配规则
            pattern = r'^image_.*\.(tif)$'
            # 只拿到第一张图像进行分析
            for batch_image in sorted(os.listdir(batch_folder_name)):
                if re.match(pattern, batch_image):
                    shutil.copyfile(os.path.join(batch_original_path, batch_image),
                                    os.path.join(output_dir, file_pre_new_name + "BF.tif"))
                    break


# 调用函数并传入根目录和输出目录
root_directory = 'D:\data\FRET_BF_image'
output_directory = 'D:\data\image'
curr_datatime = '20240515'
rename_and_copy_images(root_directory, output_directory, datatime=curr_datatime, cell_line="Mcf7")
