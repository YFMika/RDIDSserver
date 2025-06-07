from flask import Flask, request, jsonify
from src.ConnectDB import con_mysql
from src.user import User  
import bcrypt

app = Flask(__name__)

def update_user_info(username: str, **kwargs):
    """通用用户信息更新函数（支持空字符串删除属性）"""
    user = User.get_by_username(username)
    if not user:
        print(f"[ERROR] 用户 {username} 不存在")
        return False, "用户不存在", None

    update_params = {}
    sql_parts = []
    
    for key, value in kwargs.items():
        # 处理空字符串：转换为NULL
        if value == "":
            value = None  # 设置为NULL，表示删除该属性值
        update_params[key] = value
        sql_parts.append(f"{key} = %s")

    if not sql_parts:
        return False, "未提供有效更新字段", None  # 过滤掉全空的更新请求

    sql = f"UPDATE user SET {', '.join(sql_parts)} WHERE username = %s"
    params = tuple(update_params.values()) + (username,)

    try:
        cursor = con_mysql(sql, params)
        if cursor.rowcount > 0:
            print(f"[INFO] 用户 {username} 信息更新成功")
            updated_user = User.get_by_username(username)
            # 将数据库中的NULL转换为空字符串返回给客户端
            return True, "更新成功", {
                "username": updated_user.username,
                "name": updated_user.name or "",
                "gender": updated_user.gender or "",
                "phone": updated_user.phone or ""
            }
        else:
            print(f"[WARNING] 用户 {username} 信息未变更")
            return False, "信息未变更", None
    except Exception as e:
        print(f"[ERROR] 更新用户信息失败: {str(e)}")
        return False, f"更新失败: {str(e)}", None

@app.route("/uc/update_user_info", methods=["POST"])
def update_user_info_api():
    data = request.json
    username = data.get("username")
    new_name = data.get("name", "")  # 显式处理为空字符串的情况
    new_gender = data.get("gender", "")
    new_phone = data.get("phone", "")

    update_data = {
        "name": new_name if new_name is not None else "",  # 确保空值正确传递
        "gender": new_gender if new_gender is not None else "",
        "phone": new_phone if new_phone is not None else ""
    }

    # 过滤掉所有字段均为空的请求
    if all(value == "" for value in update_data.values()):
        print("[ERROR] 未提供有效更新字段（传入全空值）")
        return jsonify({"code": 400, "message": "未提供有效更新字段"}), 400

    success, message, user_data = update_user_info(username, **update_data)

    if success:
        print(f"[INFO] 成功返回用户 {username} 更新后的信息给客户端")
        return jsonify({
            "code": 200,
            "message": message,
            "data": user_data  # 包含处理后的空字符串
        })
    else:
        print(f"[ERROR] 返回给客户端用户 {username} 更新失败信息: {message}")
        return jsonify({"code": 400, "message": message}), 400


def update_password(username: str, new_password: str) -> tuple:
    """更新用户密码

    Args:
        username (str): 用户名（必填）
        new_password (str): 新密码（必填）

    Returns:
        tuple: (成功标志, 消息字符串)
            - 成功: (True, "密码更新成功")
            - 失败: (False, "错误信息")

    Raises:
        ValueError: 当参数缺失或格式不合法时抛出
    """
    # 参数校验
    if not username or not new_password:
        raise ValueError("用户名和新密码不能为空")

    # 查询用户
    user = User.get_by_username(username)
    if not user:
        return False, "用户不存在"

    # 生成新密码哈希
    salt = bcrypt.gensalt()
    hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)

    # 执行更新
    query = """
    UPDATE user 
    SET password = %s 
    WHERE username = %s
    """
    params = (hashed_new_password.decode('utf-8'), username)

    try:
        cursor = con_mysql(query, params)
        if cursor.rowcount > 0:
            return True, "密码更新成功"
        else:
            return False, "密码未变更"
    except Exception as e:
        return False, f"密码更新失败: {str(e)}"


@app.route("/uc/update_password", methods=["POST"])
def update_password_api():
    """用户密码更新接口

    请求格式:
        POST /uc/update_password
        {
            "username": "test_user",   # 用户名（必填）
            "new_password": "new_pwd"  # 新密码（必填）
        }

    返回格式:
        成功:
            {
                "code": 200,
                "message": "密码更新成功"
            }
        失败:
            {
                "code": 400,
                "message": "错误信息"
            }
    """
    try:
        data = request.json
        username = data.get("username")
        new_password = data.get("new_password")

        # 参数校验
        if not username or not new_password:
            raise ValueError("用户名和新密码不能为空")

        # 执行密码更新
        success, message = update_password(username, new_password)

        if success:
            print(f"[INFO] 成功返回用户 {username} 密码更新成功信息给客户端")
            return jsonify({"code": 200, "message": message}), 200
        else:
            print(f"[ERROR] 返回给客户端用户 {username} 密码更新失败信息: {message}")
            return jsonify({"code": 400, "message": message}), 400

    except ValueError as ve:
        print(f"[ERROR] 返回给客户端参数错误信息: {str(ve)}")
        return jsonify({"code": 400, "message": str(ve)}), 400
    except Exception as e:
        print(f"[ERROR] 返回给客户端服务器内部错误信息: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"服务器内部错误: {str(e)}"
        }), 500
    
    
def get_user_info(username: str) -> tuple:
    """
    根据用户名查询用户信息
    :param username: 用户名（必填）
    :return: 查询结果、消息及用户数据（用户不存在时数据为空字典）
    """
    user = User.get_by_username(username)
    
    if not user:
        print(f"[ERROR] 用户 {username} 不存在")
        return False, "用户不存在", {}  # 返回空字典而非None
    
    user_data = {
        "name": user.name if user.name else "",  
        "gender": user.gender if user.gender else "",
        "phone": user.phone if user.phone else ""
    }
    print(f"[INFO] 成功获取用户 {username} 的信息")
    return True, "查询成功", user_data


@app.route("/uc/get_user_info", methods=["POST"])
def get_user_info_api():
    """
    用户信息查询接口
    请求格式:
        POST /uc/get_user_info
        {
            "username": "test_user"  # 用户名（必填）
        }
    
    返回格式:
        成功:
            {
                "code": 200,
                "message": "查询成功",
                "data": {
                    "name": "用户姓名",
                    "gender": "男/女",
                    "phone": "手机号码"
                }
            }
        失败:
            {
                "code": 400,
                "message": "错误信息"
            }
    """
    try:
        data = request.json
        username = data.get("username")
        
        # 参数校验
        if not username:
            raise ValueError("用户名不能为空")
        
        # 执行查询
        success, message, user_data = get_user_info(username)
        
        if success:
            print(f"[INFO] 成功返回用户 {username} 的信息给客户端")
            return jsonify({
                "code": 200,
                "message": message,
                "data": user_data
            }), 200
        else:
            print(f"[ERROR] 返回给客户端用户 {username} 查询失败信息: {message}")
            return jsonify({"code": 400, "message": message}), 400
            
    except ValueError as ve:
        print(f"[ERROR] 返回给客户端参数错误信息: {str(ve)}")
        return jsonify({"code": 400, "message": str(ve)}), 400
    except Exception as e:
        print(f"[ERROR] 返回给客户端服务器内部错误信息: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"服务器内部错误: {str(e)}"
        }), 500