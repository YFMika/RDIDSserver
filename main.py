import os
from flask import Flask
from flask_cors import CORS
from waitress import serve
from src.RegAndLog import register, login, load_team
from src.ImageProcessing import image_processing_endpoint, confirm_save_endpoint, get_result_image, get_user_result, get_team_results, get_user_results_time, get_team_results_time
from src.UserCenter import update_password_api, update_user_info_api, get_user_info_api
# from src.Area import get_team_area
from src.TeamHandling import create_team, join_team, show_team_info, esc_team, disband_team, get_team_member, upload_team_area, get_team_area, get_team_name
import threading
import time
import socket

# 主应用实例
app = Flask(__name__)
CORS(app)  # 启用跨域资源共享

# 服务器配置
SERVER_PORT = 5000  # 服务监听端口
SERVER_HOST = "0.0.0.0"  # 监听所有可用网络接口

def print_running_status() -> None:
    """周期性打印服务器运行状态
    
    每5秒输出一次服务器当前时间，用于监控服务存活状态
    """
    while True:
        print(f"[INFO] 服务器运行中... 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(5)

# 全局标志：状态监控线程是否已启动
status_thread_started = False

# 注册相关路由
app.route("/auth/register", methods=["POST"])(register)
app.route("/auth/login", methods=["POST"])(login)
app.route("/auth/load_team", methods=["POST"])(load_team)

app.route("/ip/image_process", methods=["POST"])(image_processing_endpoint)
app.route("/ip/confirm_save", methods=["POST"])(confirm_save_endpoint)
app.route("/ip/get_result_image", methods=["GET", "POST"])(get_result_image)
app.route("/ip/get_user_results", methods=["GET", "POST"])(get_user_result)
app.route("/ip/get_team_results", methods=["GET", "POST"])(get_team_results)
app.route("/ip/get_user_results_time", methods=["POST"])(get_user_results_time)
app.route("/ip/get_team_results_time", methods=["POST"])(get_team_results_time)

app.route("/uc/update_user_info", methods=["POST"])(update_user_info_api)
app.route("/uc/update_password", methods=["POST"])(update_password_api)
app.route("/uc/get_user_info", methods=["POST"])(get_user_info_api)

app.route("/th/create_team", methods=["POST"])(create_team)
app.route("/th/join_team", methods=["POST"])(join_team)
app.route("/th/show_team_info", methods=["GET", "POST"])(show_team_info)
app.route("/th/esc_team", methods=["POST"])(esc_team)
app.route("/th/disband_team", methods=["POST"])(disband_team)
app.route("/th/get_team_member", methods=["POST"])(get_team_member)
app.route("/th/upload_team_area", methods=["POST"])(upload_team_area)
app.route("/th/get_team_area", methods=["POST"])(get_team_area)
app.route("/th/get_team_name", methods=["POST"])(get_team_name)

# app.route("/ad/get_team_area", methods=["POST"])(get_team_area)


if __name__ == "__main__":
    def start_status_thread() -> None:
        """启动服务器状态监控后台线程
        
        使用守护线程周期性打印服务器运行状态，避免阻塞主进程
        """
        global status_thread_started  # 声明使用全局变量
        if not status_thread_started:
            thread = threading.Thread(target=print_running_status, daemon=True)
            thread.start()
            status_thread_started = True
    
    # 生产环境启动逻辑
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_status_thread()
    
    # 获取本地IP地址（用于日志输出）
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    
    print(f"服务器已启动，访问地址: http://{local_ip}:{SERVER_PORT}")
    serve(app, host=SERVER_HOST, port=SERVER_PORT, threads=16)
    