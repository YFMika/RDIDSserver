import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import quote
import logging
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crawler.log'
)
logger = logging.getLogger(__name__)


class RoadDefectImageCrawler:
    def __init__(self, query_keywords=None, save_dir='road_defect_images', max_workers=5):
        """
        初始化爬虫类

        Args:
            query_keywords: 搜索关键词列表
            save_dir: 保存图片的目录
            max_workers: 线程池最大工作线程数
        """
        self.query_keywords = query_keywords or [
            "道路裂缝", "路面坑洞", "道路破损", "路面隆起",
            "road crack", "pothole", "road damage", "road surface defect"
        ]
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_search_results(self, keyword, page=1):
        """获取搜索结果页面"""
        try:
            # 使用Google图片搜索（需要科学上网）
            # encoded_keyword = quote(keyword)
            # url = f"https://www.google.com/search?q={encoded_keyword}&tbm=isch&start={str((page-1)*20)}"

            # 使用Bing图片搜索
            encoded_keyword = quote(keyword)
            url = f"https://cn.bing.com/images/search?q={encoded_keyword}&first={str((page - 1) * 35 + 1)}"

            logger.info(f"正在请求搜索页面: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"请求搜索页面失败: {e}")
            return None

    def parse_image_urls(self, html_content):
        """从HTML内容中解析图片URL"""
        if not html_content:
            return []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            image_urls = []

            # Bing图片搜索结果解析
            for item in soup.select('.iusc'):
                try:
                    m = item.get('m')
                    if m:
                        # 提取图片URL
                        import json
                        img_data = json.loads(m)
                        img_url = img_data.get('murl')
                        if img_url:
                            image_urls.append(img_url)
                except Exception as e:
                    logger.error(f"解析图片URL失败: {e}")

            # Google图片搜索结果解析（如果使用Google）
            # for img in soup.select('img'):
            #     src = img.get('src')
            #     if src and src.startswith('http'):
            #         image_urls.append(src)

            logger.info(f"解析到 {len(image_urls)} 个图片URL")
            return image_urls
        except Exception as e:
            logger.error(f"解析HTML内容失败: {e}")
            return []

    def download_image(self, url, save_path):
        """下载单个图片"""
        try:
            if os.path.exists(save_path):
                logger.info(f"图片已存在: {save_path}")
                return True

            logger.info(f"开始下载图片: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"图片下载成功: {save_path}")
            return True
        except requests.RequestException as e:
            logger.error(f"下载图片失败: {url}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"处理图片时出错: {url}, 错误: {e}")
            return False

    def process_keyword(self, keyword, max_pages=3, max_images_per_page=30):
        """处理单个关键词的搜索和下载"""
        keyword_dir = os.path.join(self.save_dir, keyword)
        if not os.path.exists(keyword_dir):
            os.makedirs(keyword_dir)

        downloaded_count = 0

        for page in range(1, max_pages + 1):
            # 获取搜索结果
            html_content = self.get_search_results(keyword, page)
            if not html_content:
                continue

            # 解析图片URL
            image_urls = self.parse_image_urls(html_content)
            if not image_urls:
                continue

            # 限制每页下载数量
            image_urls = image_urls[:max_images_per_page]

            # 下载图片
            for i, url in enumerate(image_urls):
                try:
                    # 生成保存路径
                    file_ext = os.path.splitext(url.split('?')[0])[1]
                    if not file_ext or len(file_ext) > 5:
                        file_ext = '.jpg'
                    save_path = os.path.join(keyword_dir, f"{keyword}_{page}_{i + 1}{file_ext}")

                    # 下载图片
                    if self.download_image(url, save_path):
                        downloaded_count += 1

                    # 随机延时，避免过快请求
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    logger.error(f"处理图片URL时出错: {url}, 错误: {e}")
                    continue

            # 页间延时
            time.sleep(random.uniform(3, 6))

        logger.info(f"关键词 '{keyword}' 共下载 {downloaded_count} 张图片")
        return downloaded_count

    def run(self, max_pages=3, max_images_per_page=30):
        """运行爬虫"""
        total_downloaded = 0

        logger.info(f"开始爬取道路缺陷图片，关键词数量: {len(self.query_keywords)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for keyword in self.query_keywords:
                futures.append(executor.submit(
                    self.process_keyword, keyword, max_pages, max_images_per_page
                ))

            # 获取结果
            for future in futures:
                total_downloaded += future.result()
                print(f"已经下载{total_downloaded}")

        logger.info(f"爬虫运行完成，共下载 {total_downloaded} 张图片")
        return total_downloaded


if __name__ == "__main__":
    # 创建爬虫实例
    crawler = RoadDefectImageCrawler(
        query_keywords=[
            "路面网状裂缝", "路面车辙", "波浪形道路",
            "Mesh cracks on the road surface", "Road surface rutting", "Wavy road"
        ],
        save_dir='road_defect_images',
        max_workers=3
    )

    # 运行爬虫
    try:
        total = crawler.run(max_pages=3, max_images_per_page=30)
        print(f"爬虫运行完成，共下载 {total} 张图片")
        print(f"图片已保存到目录: {os.path.abspath(crawler.save_dir)}")
    except KeyboardInterrupt:
        print("用户中断爬虫运行")
    except Exception as e:
        print(f"爬虫运行出错: {e}")
