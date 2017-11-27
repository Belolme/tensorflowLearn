import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile(
    "datasets/flower_photos/daisy/5673551_01d1ea993e_n.jpg", 'rb').read()

with tf.Session() as sess:
    # 将图像使用 jpeg 格式编码转换成对应的三维矩阵
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 压缩图片
    resized = tf.image.resize_images(img_data, [300, 300], method=0)

    # 裁剪图片
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    central_croped = tf.image.central_crop(img_data, 0.8)

    # 图像翻转 (我们常常需要把翻转后的图片加入训练集当中，这样可以使得训练出来的模型能够识别多角度的物品)
    flipped = tf.image.flip_up_down(img_data)  # 上下翻转
    filpped = tf.image.flip_left_right(img_data)  # 左右翻转
    transpposed = tf.image.transpose_image(img_data)  # 沿对角线翻转

    # 调整色彩（可以调整的参数包括：亮度，对比度，饱和度，伽马，色相）
    adjusted = tf.image.adjust_brightness(img_data, -0.5)  # 亮度的调整
    adjusted = tf.image.random_brightness(img_data, 10.0)
    adjusted = tf.image.adjust_hue(img_data, 0.1)

    # 处理标注框
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, tf.float32), 0)  # 在 0 维上多加一维变成四维矩阵

    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)

    # use pyplot tool view the image
    plt.imshow(result[0].eval())
    plt.show()
