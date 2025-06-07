import os
import tos
from PIL import Image
from tos import HttpMethodType
from volcenginesdkarkruntime import Ark

# 从环境变量获取 Access Key/API Key信息
ak = "AKLTMzhlODk2NDg1MzEyNGViNmIyZThhODcwMWRiMmE5ZDY"
sk = "TURCa1lXVTBNelJtT1RnMk5EVmpZV0ZtTnpaaU9UWmlNMkUxWXpkbVpXSQ=="
api_key = "f1f7b9f0-cfd5-4987-b418-d4e5d9cdc785"

# 替换 <MODEL> 为模型的Model ID
model_id="doubao-1.5-thinking-vision-pro-250428"

# 压缩前图片
original_file = "test.jpg"
# 压缩后图片存放路径
compressed_file = "comressed_image.jpg"
# 压缩的目标图片大小，300KB
target_size = 300 * 1024

# endpoint 和 region 填写Bucket 所在区域对应的Endpoint。
# 以华北2(北京)为例，region 填写 cn-beijing。
# 公网域名endpoint 填写 tos-cn-beijing.volces.com
endpoint, region = "tos-cn-beijing.volces.com", "cn-beijing"
# 对象桶名称
bucket_name = "qw4231123"
# 对象名称，例如 images 下的 compressed_image.jpeg 文件，则填写为 images/compressed_image.jpeg
object_key = "test.jpg"

def compress_image(input_path, output_path):
    img = Image.open(input_path)
    current_size = os.path.getsize(input_path)

    # 粗略的估计压缩质量，也可以从常量开始，逐步减小压缩质量，直到文件大小小于目标大小
    image_quality = int(float(target_size / current_size) * 100)
    img.save(output_path, optimize=True, quality=int(float(target_size / current_size) * 100))

    # 如果压缩后文件大小仍然大于目标大小，则继续压缩
    # 压缩质量递减，直到文件大小小于目标大小
    while os.path.getsize(output_path) > target_size:
        img = Image.open(output_path)
        image_quality -= 10
        if image_quality <= 0:
            break
        img.save(output_path, optimize=True, quality=image_quality)
    return image_quality

def upload_tos(filename, tos_endpoint, tos_region, tos_bucket_name, tos_object_key):
    # 创建 TosClientV2 对象，对桶和对象的操作都通过 TosClientV2 实现
    tos_client, inner_tos_client = tos.TosClientV2(ak, sk, tos_endpoint, tos_region), tos.TosClientV2(ak, sk,
                                                                                                      tos_endpoint,
                                                                                                      tos_region)
    try:
        # 将本地文件上传到目标桶中, filename为本地压缩后图片的完整路径
        tos_client.put_object_from_file(tos_bucket_name, tos_object_key, filename)
        # 获取上传后预签名的 url
        return inner_tos_client.pre_signed_url(HttpMethodType.Http_Method_Get, tos_bucket_name, tos_object_key)
    except Exception as e:
        if isinstance(e, tos.exceptions.TosClientError):
            # 操作失败，捕获客户端异常，一般情况为非法请求参数或网络异常
            print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
        elif isinstance(e, tos.exceptions.TosServerError):
            # 操作失败，捕获服务端异常，可从返回信息中获取详细错误信息
            print('fail with server error, code: {}'.format(e.code))
            # request id 可定位具体问题，强烈建议日志中保存
            print('error with request id: {}'.format(e.request_id))
            print('error with message: {}'.format(e.message))
            print('error with http code: {}'.format(e.status_code))
        else:
            print('fail with unknown error: {}'.format(e))
        raise e


if __name__ == "__main__":
    print("----- 压缩图片 -----")
    quality = compress_image(original_file, compressed_file)
    print("Compressed Image Quality: {}".format(quality))

    print("----- 上传至TOS -----")
    pre_signed_url_output = upload_tos(compressed_file, endpoint, region, bucket_name, object_key)
    print("Pre-signed TOS URL: {}".format(pre_signed_url_output.signed_url))

    print("----- 传入图片调用视觉理解模型 -----")
    client = Ark(api_key=api_key)

    # 图片输入:
    response = client.chat.completions.create(
        # 配置推理接入点
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pre_signed_url_output.signed_url
                        }
                    },
                    {"type": "text", "text": "我有五类缺陷和4类严重程度，缺陷有裂缝、网状裂缝、表面损害类（松散、坑槽、泛油等）、修补类缺陷、"
                                             "变形类缺陷（车辙、波浪形等），标号分别为1~5，严重程度有0~3，0代表没有缺陷。"
                                             "下面的图片的缺陷种类（可能多种）和严重程度是什么，返回如下的json格式{type: [repair],severity: [2]}}",}
                ],
            }
        ],
    )

    print(response.choices[0])