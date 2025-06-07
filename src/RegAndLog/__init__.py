from flask import Flask, request, jsonify
from src.ConnectDB import con_mysql  
from src.user import User
from src.team import Team
from src.team_member import Team_member
import bcrypt

app = Flask(__name__)


def insert_user(username: str, password: str, user_type: int) -> tuple:
    """插入新用户，密码自动加密"""
    # 先查询用户是否存在
    if User.get_by_username(username):
        print(f"[WARNING] 注册失败: 用户名 '{username}' 已存在")
        return False, "用户名已存在"
        
    # 生成盐并哈希密码
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    query = """
    INSERT INTO user (username, password, user_type)
    VALUES (%s, %s, %s)
    """
    
    try:
        con_mysql(query, (username, hashed.decode('utf-8'), user_type))
        return True, "注册成功"
    except Exception as e:
        print(f"[ERROR] 注册失败: {str(e)}")  # 新增注册失败日志
        return False, f"注册失败: {str(e)}"
    

@app.route("/auth/register", methods=["POST"])
def register():
    """
    用户注册接口
    
    接收用户名、密码及可选用户类型，验证唯一性后创建新用户。
    
    Request Body:
        {
            "username": str,   # 用户名（必填，唯一标识）
            "password": str,   # 密码（必填，建议8-20位复杂字符）
            "user_type": str   # 用户类型（可选，默认值为"user"，可选值如"admin"）
        }
    
    Returns:
        JSON响应:
        {
            "code": int,       # 状态码（200=成功，400=参数错误/注册失败，500=服务器错误）
            "message": str     # 操作结果描述
        }
    
    处理流程:
        1. 解析请求参数（用户名、密码、用户类型）
        2. 调用insert_user函数执行注册逻辑（包含唯一性校验和密码哈希处理）
        3. 根据返回结果返回成功或失败响应
    
    错误处理:
        - 400: 用户名已存在、密码格式错误、参数缺失
        - 500: 数据库操作失败
    
    安全说明:
        - 密码在insert_user函数中会使用bcrypt进行哈希处理，禁止明文存储
    """
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user_type = data.get("user_type")  
    
    success, message = insert_user(username, password, user_type)
    if success:
        print(f"[INFO] 成功返回注册成功信息给客户端: {username}")  # 新增成功日志
        return jsonify({"code": 200, "message": message})
    else:
        print(f"[ERROR] 返回注册失败信息给客户端: {message}")  # 新增失败日志
        return jsonify({"code": 400, "message": message}), 400
    

@app.route("/auth/login", methods=["POST"])
def login():
    """
    用户登录认证接口
    
    接收用户名、密码及可选用户类型，验证用户身份和权限后，
    返回用户基本信息及所属小组ID（若存在）。
    
    Request Body:
        {
            "username": str,   # 用户名
            "password": str,   # 密码
            "user_type": str   # 用户类型
        }
    
    Returns:
        JSON响应:
        {
            "code": int,               # 状态码（200=成功，401=认证失败，404=用户不存在，500=服务器错误）
            "message": str,            # 状态信息
            "data": {                  # 登录用户信息（仅成功时返回）
                "user_id": int,        # 用户ID
                "username": str,       # 用户名
                "user_type": str,      # 用户类型（如"admin"、"user"）
                "user_team": int      # 所属小组ID（无小组时为-1）
            }
        }
    
    处理流程:
        1. 解析请求参数（用户名、密码、用户类型）
        2. 根据用户名查询用户记录
        3. 验证用户是否存在
        4. 校验密码正确性（使用bcrypt哈希对比）
        5. 可选校验用户类型（若请求中包含user_type参数）
        6. 查询用户所属小组（返回首个小组ID，无小组时为-1）
        7. 返回登录结果及用户信息
    
    错误处理:
        - 404: 用户不存在
        - 401: 密码错误或用户类型不匹配
        - 500: 数据库查询失败或其他异常
    
    安全说明:
        - 密码存储使用bcrypt哈希，禁止明文传输和存储
        - 用户类型校验为可选逻辑，需根据业务需求决定是否启用
        - 小组信息返回用户所属的第一个小组，建议根据业务场景调整查询逻辑
    
    注意事项:
        - 响应中的user_team为小组ID，-1表示用户未加入任何小组
        - 生产环境建议添加登录频率限制和失败次数监控
        - 敏感信息（如密码）需通过HTTPS传输
    """
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user_type = data.get("user_type")
    
    user = User.get_by_username(username)
    if not user:
        print(f"[ERROR] 登录失败: 用户 '{username}' 不存在")  # 新增用户不存在日志
        return jsonify({"code": 404, "message": "用户不存在"}), 404
    
    # 验证密码失败
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        print(f"[ERROR] 登录失败: 用户名 '{username}' 密码错误")  # 新增密码错误日志
        return jsonify({"code": 401, "message": "密码错误"}), 401
    
    # 验证用户类型失败
    if user_type is not None and user_type != user.user_type:
        print(f"[ERROR] 登录失败: 用户名 '{username}' 用户类型错误")  # 新增类型错误日志
        return jsonify({"code": 401, "message": "用户类型错误"}), 401
    
    user_team = Team_member.get_teams_name_by_user(user.user_id)
    if user_team and len(user_team) > 0:
        user_team = user_team[0]
    else:
        user_team = "l1l1l1l1"
    
    # 登录成功日志
    print(f"[INFO] 成功返回登录成功信息给客户端: {username}")
    return jsonify({
        "code": 200, 
        "message": "登录成功",
        "data": {
            "user_id": user.user_id,
            "username": user.username,
            "user_type": user.user_type,
            "user_team": user_team
        }
    })


@app.route("/auth/load_team", methods=["POST"])
def load_team():
    """
    通过小组ID加载小组信息（用户登录关联小组场景）
    
    接收小组ID，验证小组存在性后返回小组详细信息，
    用于用户登录时关联所属小组的业务场景。
    
    Request Body:
        {
            "team_id": int  # 目标小组ID
        }
    
    Returns:
        JSON响应:
        {
            "code": int,               # 状态码（200=成功，404=小组不存在，500=服务器错误）
            "message": str,            # 状态信息
            "data": {                  # 小组详细信息（仅成功时返回）
                "team_id": int,        # 小组ID
                "team_name": str,      # 小组名称
                "leader_id": int,      # 组长用户ID
                "task_area": str       # 小组任务区域
            }
        }
    
    处理流程:
        1. 解析请求参数中的小组ID
        2. 根据ID查询小组信息
        3. 验证小组是否存在
        4. 返回小组信息或错误提示
    
    错误处理:
        - 404: 小组ID不存在
        - 500: 数据库查询失败或其他异常
    
    注意事项:
        - 假设调用此接口前已完成用户身份验证
        - 返回数据包含小组核心信息，可根据业务需求扩展字段
    """
    data = request.json
    team_id = data.get("team_id")
    
    team = Team.get_by_id(team_id)
    if not team:
        print(f"[ERROR] 登录失败: 小组 '{team_id}' 不存在")  # 新增用户不存在日志
        return jsonify({"code": 404, "message": "小组不存在"}), 404
    
    # 登录成功日志
    print(f"[INFO] 成功返回登录小组信息给客户端: {team_id}")
    return jsonify({
        "code": 200, 
        "message": "登录小组信息获取成功",
        "data": {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "leader_id": team.leader_id,
            "task_area": team.task_area
        }
    })
