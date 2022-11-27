import os
import paddle.fluid as fluid
import numpy as np
from PIL import Image

# 根据竞赛规则更改 TopK 的值。本次竞赛只看 top 1 的值
TOP_K = 1

DATA_DIM = 224

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# 下面行代码根据自己保存时的写法匹配
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model('models/step_2_model/', exe,
                                                                                      model_filename='model',
                                                                                      params_filename='params'
                                                                                      )


def real_infer_one_img(im):
    infer_result = exe.run(
        inference_program,
        feed={feed_target_names[0]: im},
        fetch_list=fetch_targets)

    # print(infer_result)
    # 打印预测结果
    mini_batch_result = np.argsort(infer_result)  # 找出可能性最大的列标，升序排列
    # print(mini_batch_result.shape)
    mini_batch_result = mini_batch_result[0][:, -TOP_K:]  # 把这些列标拿出来
    # print('预测结果：%s' % mini_batch_result)
    # 打印真实结果
    # label = np.array(test_y)  # 转化为 label
    # print('真实结果：%s' % label)
    mini_batch_result = mini_batch_result.flatten()  # 拉平了，只吐出一个 array
    mini_batch_result = mini_batch_result[::-1]  # 逆序
    return mini_batch_result


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def process_image(img_path):
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype(np.float32).transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    img = np.expand_dims(img, axis=0)
    return img


def convert_list(my_list):
    my_list = list(my_list)
    my_list = map(lambda x: str(x), my_list)
    # print('_'.join(my_list))
    return '_'.join(my_list)


def infer(file_path):
    im = process_image(file_path)
    result = real_infer_one_img(im)
    result = convert_list(result)
    return result


def createCSVFile(cat_12_test_path):
    lines = []

    # 获取所有的文件名
    img_paths = os.listdir(cat_12_test_path)
    for file_name in img_paths:
        file_name = file_name
        file_abs_path = os.path.join(cat_12_test_path, file_name)
        result_classes = infer(file_abs_path)

        file_predict_classes = result_classes

        line = '%s,%s\n' % (file_name, file_predict_classes)
        lines.append(line)

    with open('result.csv', 'w') as f:
        f.writelines(lines)


abs_path = r'/cat_12_test'  # cat_12_test 文件夹的真实路径
createCSVFile(abs_path)
