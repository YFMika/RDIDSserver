from flask import Flask, request, jsonify
from src.user import User  
from src.team import Team
from src.team_member import Team_member
from src.ConnectDB import con_mysql
import pymysql

app = Flask(__name__)


def get_team_name_by_user_id(username: str):
    team_names = Team_member.get_teams_name_by_user(username)
    return team_names[0] if team_names else None


def get_team_info(team_name: str):
    """
    内部方法：获取指定小组的所有成员详细信息
    
    参数:
        team_name (str): 小组名称
    
    Returns:
        JSON响应包含组员的username、name、gender和user_type和area_name
    """
    try:
        if team_name is None:
            print(f"[WARNING] 展示失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 404, "message": "小组不存在"}), 404
        
        # 使用Team类查询小组
        team = Team.get_by_name(team_name)
        if not team:
            print(f"[WARNING] 展示失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 404, "message": "小组不存在"}), 404
        
        try:
            # 使用TeamMember类获取所有组员的user_id
            user_ids = Team_member.get_members_by_team(team.team_id)

            if not user_ids:
                print(f"[WARNING] 展示失败: 小组名 '{team_name}' 中没有成员")
                return jsonify({"code": 404, "message": "小组中没有成员"}), 404
            
            # 查询字段变更，获取user_type而非phone
            user_info = con_mysql(
                "SELECT username, name, gender, user_type, user_id FROM user WHERE user_id IN ({})".format(
                    ','.join(['%s'] * len(user_ids))
                ),
                tuple(user_ids),
                fetch_all=True
            )

            if not user_info:
                print(f"[WARNING] 展示失败: 查询用户id失败")
                return jsonify({"code": 404, "message": "小组中没有成员"}), 404
            
            area_name = team.task_area

            # 处理查询结果，添加area_name字段
            result = []
            for user in user_info:
                result.append({
                    'username': user['username'],
                    'name': user['name'],
                    'gender': user['gender'],
                    'user_type': user['user_type'],
                    'area_name': area_name
                })
            
            print(f"[INFO] 成功返回小组成员信息给客户端: 小组: '{team_name}'")
            return jsonify({
                "code": 200,
                "message": "查询成功",
                "team_name": team_name,
                "data": result
            })
            
        except pymysql.MySQLError as e:
            return jsonify({"code": 500, "message": f"数据库错误: {str(e)}"}), 500
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 展示所有小组成员失败 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500


@app.route("/th/create_team", methods=["POST"])
def create_team():
    """
    创建新小组并将指定用户设为组长
    
    接收小组名称和用户名，验证用户未在其他小组且小组名称唯一后，
    创建新小组并将用户添加为组长和首个成员。
    
    Request Body:
        {
            "team_name": str,  # 新小组名称（唯一标识）
            "username": str    # 创建者用户名
        }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400=参数错误，500=服务器错误）
            "message": str         # 操作结果描述
        }
    
    处理流程:
        1. 验证请求参数完整性
        2. 检查用户是否存在
        3. 检查用户是否已加入其他小组
        4. 检查小组名称是否已存在
        5. 创建新小组记录（设置用户为组长）
        6. 将用户添加为该小组成员
        7. 返回创建成功信息
        
    错误处理:
        - 400: 用户已在小组中 / 小组名称已存在
        - 500: 数据库操作失败 / 其他未知错误
        
    注意事项:
        - 每个用户最多只能属于一个小组
        - 小组名称全局唯一
        - 创建者自动成为组长（leader_id）
    """
    # 先查询小组是否存在和用户是否已经在某个小组中
    try:
        data = request.json
        team_name = data.get("team_name")
        username = data.get("username")
        user = User.get_by_username(username)

        if User.whether_user_in_team(user.user_id):
            print(f"[WARNING] 用户 '{username}' 已经在一个小组 '{team_name}' 中")
            return jsonify({"code": 400, "message": "用户已经在一个小组中"})

        ret = Team.get_by_name(team_name)
        if not ret:
            team = Team(team_name=team_name, leader_id=user.user_id)
            team.save()
            team_id = Team.get_by_name(team_name).team_id
            team_member = Team_member(team_id=team_id, user_id=user.user_id)
            team_member.save()
        else:
            print(f"[WARNING] 新增失败: 小组名 '{team_name}' 已存在")
            return jsonify({"code": 400, "message": "小组名已存在"}), 400
        
        print(f"[INFO] 成功返回创建成功信息给客户端: 小组: {team_name}, 用户: {username}")
        return jsonify({"code": 200, "message": "小组创建成功"})
    
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 创建小组失败 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500


@app.route("/th/join_team", methods=["POST"])
def join_team():
    """
    用户加入指定小组（需用户未在其他小组中）
    
    接收小组名称和用户名，验证用户状态和小组存在性后，
    将用户添加至目标小组，并返回小组所有成员信息。
    
    Request Body:
        {
            "team_name": str,  # 目标小组名称
            "username": str    # 申请加入的用户名
        }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400/404/500=错误）
            "message": str,        # 状态信息
            "data": list           # 小组所有成员信息（仅成功时返回）
        }
    
    处理流程:
        1. 解析请求参数（小组名称、用户名）
        2. 验证用户是否存在
        3. 检查用户是否已在其他小组中
        4. 验证小组是否存在
        5. 检查用户是否已在目标小组中
        6. 将用户添加至目标小组（插入Team_member表）
        7. 查询并返回小组所有成员信息
    
    错误处理:
        - 400: 用户已在其他小组 / 用户已在目标小组 / 小组不存在 / 参数缺失
        - 404: 小组不存在
        - 500: 数据库操作失败
    
    注意事项:
        - 每个用户同一时间只能属于一个小组
        - 加入小组前需确保小组存在且用户未加入过
        - 返回的成员信息包含：username、name、gender、phone
    """
    # 先查询小组是否存在和用户是否已经在某个小组中
    try:
        data = request.json
        team_name = data.get("team_name")
        username = data.get("username")
        team = Team.get_by_name(team_name)

        user = User.get_by_username(username)
        user_id = user.user_id
        if User.whether_user_in_team(user.user_id):
            print(f"[WARNING] 用户 '{username}' 已经在一个小组 '{team_name}' 中")
            return jsonify({"code": 400, "message": "用户已经在一个小组中"})

        if team:
            team_id = team.team_id
            if Team_member.get_by_team_and_user(team_id, user_id):
                print(f"[WARNING] 加入失败: 用户 '{username}' 已在小组 '{team_name}' 中")
                return jsonify({"code": 400, "message": "该用户已经在该小组中"}), 400
            else:
                team_member = Team_member(team_id, user_id)
                team_member.save()
                user_ids = Team_member.get_members_by_team(team_id)

                user_info = con_mysql(
                    "SELECT username, name, gender, phone FROM user WHERE user_id IN ({})".format(
                        ','.join(['%s'] * len(user_ids))
                    ),
                    tuple(user_ids),
                    fetch_all=True
                )
                
                print(f"[INFO] 成功返回加入成功信息给客户端: 小组: '{team_name}', 用户: '{username}'")
                return jsonify({
                    "code": 200,
                    "message": "小组加入成功",
                    "data": user_info
                })

        else:
            print(f"[WARNING] 注册失败: 用户名 '{username}' 已存在")
            return jsonify({"code": 400, "message": "用户名已存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 加入小组失败 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500


@app.route("/th/show_team_info", methods=["GET", "POST"])
def show_team_info():
    """
    获取指定小组的所有成员详细信息
    
    支持通过GET参数或POST JSON获取小组名称，返回包含所有成员信息的JSON响应。
    
    Request:
        GET请求:
            - 查询参数:
                username (str): 用户名称
                
        POST请求:
            - JSON数据:
                {
                    "username": str  # 用户名称
                }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400/404/500=错误）
            "message": str,        # 状态信息
            "data": [
                {
                    "username": str,    # 用户名
                    "name": str,        # 姓名（若不存在则为None）
                    "gender": str,      # 性别（若不存在则为None）
                    "user_type": str,   # 用户类型
                    "area_name": int      # 区域ID（若不存在则为None）
                },
                ...
            ]
        }
    
    处理流程:
        1. 根据请求方法获取小组名称
        2. 验证小组是否存在
        3. 获取小组成员的user_id列表
        4. 查询用户基本信息（用户名、姓名、性别、用户类型）
        5. 查询用户区域信息（area_name）
        6. 合并数据并返回响应
        
    注意事项:
        - 支持两种请求方式（GET/POST）
        - 用户区域信息不存在时返回None
    """
    try:
        # 支持GET和POST两种方式获取参数
        if request.method == 'GET':
            user_id = request.args.get("user_id")
        else:
            data = request.json
            user_id = data.get("user_id") if data else None

        team_name = get_team_name_by_user_id(user_id)
        return get_team_info(team_name)

    except Exception as e:
        print(f"[ERROR] 获取参数失败 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500
    

@app.route("/th/esc_team", methods=["POST"])
def esc_team():
    """
    用户退出指定小组
    
    接收小组名称和用户名，验证用户是否为小组成员，
    若是则将用户从小组中移除，否则返回错误信息。
    
    Request Body:
        {
            "team_name": str,  # 目标小组名称
            "username": str     # 要退出的用户名称
        }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400/404/500=错误）
            "message": str         # 状态信息
        }
    
    处理流程:
        1. 验证小组是否存在
        2. 验证用户是否存在
        3. 检查用户是否为小组成员
        4. 执行退出操作（删除用户与小组的关联）
        5. 返回操作结果
        
    错误处理:
        - 400: 小组不存在、用户不在小组中、参数缺失
        - 404: 小组不存在
        - 500: 数据库操作失败
    """
    # 先查询小组是否存在
    try:
        data = request.json
        team_name = data.get("team_name")
        username = data.get("username")
        team = Team.get_by_name(team_name)
        if team:
            team_id = team.team_id
            user_id = User.get_by_username(username).user_id

            if not Team_member.get_by_team_and_user(team_id, user_id):
                print(f"[WARNING] 退出失败: 用户 '{username}' 不在小组 '{team_name}' 中")
                return jsonify({"code": 400, "message": "该用户不在该小组中"}), 400
            else:
                team_member = Team_member(team_id, user_id)
                team_member.delete()
                print(f"[INFO] 退出成功: 用户 '{username}' 成功退出小组 '{team_name}'")
                return jsonify({"code": 200, "message": "退出小组成功"})

        else:
            print(f"[WARNING] 退出失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 400, "message": "小组不存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 退出小组出错 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500


@app.route("/th/disband_team", methods=["POST"])
def disband_team():
    """
    解散指定小组（需管理员权限）
    
    接收小组名称和操作用户名，验证用户是否为小组管理员，
    若是则删除整个小组及其相关数据，否则返回权限错误。
    
    Request Body:
        {
            "team_name": str,  # 待解散的小组名称
            "username": str    # 执行解散操作的用户名
        }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400/404/500=错误）
            "message": str         # 状态信息
        }
    
    处理流程:
        1. 验证小组是否存在
        2. 验证用户是否存在
        3. 检查用户是否为小组管理员
        4. 执行小组解散操作（删除小组及关联数据）
        5. 返回操作结果
        
    错误处理:
        - 400: 小组不存在、用户不是管理员、参数缺失
        - 404: 小组不存在
        - 500: 数据库操作失败
    """
    # 先查询小组是否存在
    try:
        data = request.json
        team_name = data.get("team_name")
        username = data.get("username")
        team = Team.get_by_name(team_name)
        if team:
            team_id = team.team_id
            user_id = User.get_by_username(username).user_id

            if not Team_member.get_by_team_and_user(team_id, user_id):
                print(f"[WARNING] 解散失败: 用户 '{username}' 不在小组 '{team_name}' 中")
                return jsonify({"code": 400, "message": "该用户不是该小组的管理者"}), 400
            else:
                team_member = Team_member(team_id, user_id)
                team_member.disband()
                print(f"[INFO] 解散成功: 用户 '{username}' 成功解散小组 '{team_name}'")
                return jsonify({"code": 200, "message": "解散小组成功"})

        else:
            print(f"[WARNING] 解散失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 400, "message": "小组不存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库解散小组失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 解散小组出错 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500
    

@app.route("/th/get_team_member", methods=["POST"])
def get_team_member():
    """
    获取小组的成员数量及组员列表
    
    接收包含 team_name 的 JSON，返回成员数量及不含管理员的 user_id 列表
    
    Request Body:
        {
            "team_name": int  # 小组名字
        }
    
    Returns:
        JSON响应:
        {
            "code": int,           # 状态码（200=成功，400/404/500=错误）
            "message": str,        # 状态信息
            "data": {
                "count": int,          # 成员数量
                "member_user_ids": list  # 组员user_id列表
            }
        }
    """
    try:
        # 解析请求参数
        data = request.json
        team_name = data.get("team_name")
        
        if not team_name:
            return jsonify({
                "code": 400,
                "message": "缺少必需参数 team_name",
                "data": {"count": 0, "member_user_ids": []}
            }), 400
        
        # 查询小组信息
        team = Team.get_by_name(team_name)
        if not team:
            return jsonify({
                "code": 404,
                "message": "小组不存在",
                "data": {"count": 0, "member_user_ids": []}
            }), 404
        
        # 获取所有成员ID（包括管理员）
        all_member_ids = Team_member.get_members_by_team(team.team_id)
        if not all_member_ids:
            return jsonify({
                "code": 200,
                "message": "小组无成员",
                "data": {"count": 0, "member_user_ids": []}
            }), 200
        
        admin_id = team.leader_id  # 管理员ID
        
        # 过滤掉管理员ID
        valid_member_ids = [user_id for user_id in all_member_ids if user_id != admin_id]
        member_count = len(valid_member_ids)
        
        print(f"[INFO] 小组名字 {team_name} 有效成员数：{member_count}，成员ID：{valid_member_ids}")
        
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": {
                "count": member_count,
                "member_user_ids": valid_member_ids
            }
        }), 200
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库错误: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"数据库操作失败: {str(e)}",
            "data": {"count": 0, "member_user_ids": []}
        }), 500
    except Exception as e:
        print(f"[ERROR] 服务器错误: {str(e)}")
        return jsonify({
            "code": 500,
            "message": "服务器内部错误，请稍后重试",
            "data": {"count": 0, "member_user_ids": []}
        }), 500
    
    
@app.route("/th/upload_team_area", methods=["POST"])
def upload_team_area():
    try:
        # 解析请求参数
        data = request.json
        user_id = data.get("user_id")
        area_name = data.get("area_name")
        team_boundary = data.get("team_boundary")

        team_name = get_team_name_by_user_id(user_id)
        
        team = Team.get_by_name(team_name)
            
        if team:
            team.task_area = area_name
            team.area_bound = team_boundary
            
            team.save()
            print(f"[INFO] 上传区域成功: 小组 '{team_name}' 成功上传区域 '{area_name}'")
            return jsonify({"code": 200, "message": "解散小组成功"})

        else:
            print(f"[WARNING] 上传区域失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 400, "message": "小组不存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库上传区域失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 上传区域出错 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500
    

@app.route("/th/get_team_area", methods=["POST"])
def get_team_area():
    try:
        # 解析请求参数
        data = request.json
        user_id = data.get("user_id")

        team_name = get_team_name_by_user_id(user_id)
            
        team = Team.get_boundary(team_name)

        if team:
            print(f"[INFO] 获取区域成功: 小组 '{team_name}' 成功获取区域 '{team.task_area}'")
            return jsonify({"code": 200, "message": "获取小组成功", "data": team.area_bound})

        else:
            print(f"[WARNING] 获取区域失败: 小组名 '{team_name}' 不存在")
            return jsonify({"code": 400, "message": "小组不存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库获取区域失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 获取区域出错 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500
    

@app.route("/th/get_team_name", methods=["POST"])
def get_team_name():
    try:
        # 解析请求参数
        data = request.json
        username = data.get("username")
            
        user = User.get_by_username(username)

        if user:
            team_names = Team_member.get_teams_name_by_user(user.user_id)
            if team_names:
                print(f"[INFO] 获取小组成功: 用户 '{username}' 成功获取小组 '{team_names[0]}'")
                return jsonify({"code": 200, "message": "获取小组成功", "data": team_names[0]})
            else:
                print(f"[INFO] 获取小组成功: 用户 '{username}' 没有小组")
                return jsonify({"code": 201, "message": "获取小组成功"})

        else:
            print(f"[WARNING] 获取小组失败: 用户名 '{username}' 不存在")
            return jsonify({"code": 400, "message": "用户不存在"}), 400
        
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库获取小组失败, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 获取小组出错 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"服务器内部错误，请稍后重试 {str(e)}"}), 500
    