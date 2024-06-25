"""
To randomly select 6 categories from a total of 80, and then pick 1 image from each of these categories.
"""
import argparse
import json
import os
import random
import cv2
import numpy as np
import typing as t


def parse_args():
    parser = argparse.ArgumentParser(description='random choose picture')
    parser.add_argument('--cls-dir', default=r'C:\dataset\ImageNet1k-2012\val', type=str, help='class dir')
    parser.add_argument('--save', default=r'D:\projects\mmpretrain\work_dirs\random_pic', type=str,
                        help='Store the randomly generated image path.')
    parser.add_argument('--save-joint', default=r'D:\projects\mmpretrain\work_dirs\visual_pic_joint', type=str,
                        help='The folder containing the concatenated heatmap')
    parser.add_argument('--heatmap-dir', default=r'D:\projects\mmpretrain\work_dirs\visual_pic', type=str,
                        help='heatmap dir')
    parser.add_argument('--num', default=6, type=int, help='number of pictures to choose')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    args = parser.parse_args()
    return args


def random_choose_pic(cls_dir: str, num: int, save: str) -> None:
    list_dir = os.listdir(cls_dir)
    rand_sample = random.sample(list_dir, num)
    random_paths = []
    os.makedirs(save, exist_ok=True)
    for dir_ in rand_sample:
        img_dir = os.path.join(cls_dir, dir_)
        img_files = os.listdir(img_dir)
        img_sample: str = random.sample(img_files, 1)[0]
        img_path = os.path.join(img_dir, img_sample)
        random_paths.append(img_path)
    with open(os.path.join(save, f'img-{dir_}'), 'w') as f:
        json.dump(random_paths, f)


def generate_pics():
    """
    generate random pictures
    """
    args = parse_args()
    for _ in range(args.epoch):
        random_choose_pic(args.cls_dir, args.num, args.save)


def concat_images_hstack(images: t.List[str], spacing: int = 10, background_color: t.Tuple = (255, 255, 255)):
    """
    Horizontally stitch multiple images and add specified color intervals between them.


    """
    # 确保所有图像的高度一致
    images = [cv2.imread(image) for image in images]
    max_height = max(image.shape[0] for image in images)

    # 调整图像大小并添加边距
    resized_images = []
    for image in images:
        # 调整图像高度
        scale_ratio = max_height / image.shape[0]
        resized_width = int(image.shape[1] * scale_ratio)
        resized_image = cv2.resize(image, (resized_width, max_height))

        # 添加左边距（除了第一张图）
        if resized_images:
            left_margin = np.full((max_height, spacing, 3), background_color, dtype=np.uint8)
            resized_image = np.hstack((left_margin, resized_image))

        resized_images.append(resized_image)

    # 横向拼接所有图像
    final_image = np.hstack(resized_images)
    return final_image


def combine_pics():
    """
    combine heatmap of different models
    """
    args = parse_args()
    img_paths = []
    model_vis_list = os.listdir(args.heatmap_dir)
    m_vis_len = len(model_vis_list)

    if not os.path.exists(args.save_joint):
        os.makedirs(args.save_joint)

    # assert
    base_model = os.path.join(args.heatmap_dir, model_vis_list[0])
    img_g_names = os.listdir(base_model)

    for cls_name in img_g_names:
        temp = [['' for _ in range(m_vis_len)] for _ in range(args.num)]
        for i in range(m_vis_len):
            cur_path = rf'{args.heatmap_dir}\{model_vis_list[i]}\{cls_name}'
            cur_img_paths = os.listdir(cur_path)
            for j in range(len(cur_img_paths)):
                temp[j][i] = os.path.join(cur_path, cur_img_paths[j])
        for j in range(len(temp)):
            cur_pic = concat_images_hstack(temp[j])
            cv2.imwrite(os.path.join(args.save_joint, f'{cls_name}_{j}.jpg'), cur_pic)
        print(f'{cls_name} processed successfully!')
    print(f'sequence: {model_vis_list}')


if __name__ == '__main__':
    # python .\tools\visualization\process_pic.py --cls-dir '' --save '' --epoch 100 --num 6
    # generate_pics()
    combine_pics()
