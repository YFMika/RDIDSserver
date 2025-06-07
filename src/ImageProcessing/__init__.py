from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from src.ConnectDB import con_mysql
from src.user import User
from src.team import Team
import pymysql
from rdds.test import test
from src.TeamHandling import get_team_name_by_user_id

app = Flask(__name__)

# 硬编码为指定目录（注意Windows路径使用双反斜杠或原始字符串）
UPLOAD_FOLDER = r'd:\\GitProject\\RDDS\\Server\\uploads'
# 确保目录存在，不存在则创建（兼容跨平台，自动处理路径分隔符）
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

"""
缺陷类型映射关系：
- mapping: 字符串到数值的映射
- mapping_reverse: 数值到字符串的反向映射
"""
mapping = {"lie": 0, "mesh": 1, "face": 2, "repair": 3, "Transformation": 4}
mapping_reverse = {0: "单向裂缝", 1: "网状裂缝", 2: "表面类缺陷", 3: "修复类缺陷", 4: "变形类缺陷"}


def image_processing(image_path: str) -> 'tuple[str, int]':
    """处理图像并返回缺陷类型和严重程度"""
    ret_classes, ret_severity = test(image_path=image_path)
    ret_classes = ret_classes[0]
    
    classes = ""
    for i in range(len(ret_classes)):
        if ret_classes[i] == 1:
            classes = classes + mapping_reverse[i] + " "
    if len(classes) == 0:
        classes = "正常"
    
    return classes.strip(), int(ret_severity)


@app.route("/ip/image_process", methods=['POST'])
def image_processing_endpoint():
    """图像处理接口"""
    username = request.form['user_name']
    gps_location = request.form['gps_location']
    detection_time = request.form['detection_time']
    file = request.files['image']
    filename = secure_filename(file.filename)
    
    # 硬编码路径拼接，使用os.path.join确保跨平台兼容性
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        user_data = con_mysql("SELECT user_id FROM user WHERE username = %s", (username,))
        if not user_data:
            print(f"[ERROR] 用户未找到: {username}")
            return jsonify({"code": 404, "message": f"用户未找到: {username}"}), 404
        user_id = user_data['user_id']
        print(f"[INFO] 成功获取用户ID: {user_id} 对应用户名: {username}")
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询用户失败: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    defect_type, severity = image_processing(file_path)
    print(f"[INFO] 图像处理完成 - 缺陷类型: {defect_type}, 严重程度: {severity}")

    try:
        insert_sql = """
        INSERT INTO result 
        (user_id, image_path, gps_location, detection_time, defect_type, severity)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        # 存入数据库的路径为硬编码目录+文件名（绝对路径）
        params = (user_id, file_path, gps_location, detection_time, defect_type, severity)
        cursor = con_mysql(insert_sql, params)
        result_id = cursor.lastrowid
        if cursor.rowcount != 1:
            raise pymysql.MySQLError("结果插入失败")
        print(f"[INFO] 数据库插入成功 - result_id: {result_id}, 参数: {params}")
    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库插入失败: {str(e)}, 参数: {params}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 图像处理接口异常: {str(e)}")
        return jsonify({"code": 500, "message": "服务器内部错误，请稍后重试"}), 500

    print(f"[INFO] 成功返回处理结果给客户端 - result_id: {result_id}")
    return jsonify({
        "code": 200,
        "message": "处理成功",
        "data": {
            "result_id": result_id,
            "gps_location": gps_location,
            "defect_type": defect_type,
            "severity": severity,
            "detection_time": detection_time
        }
    })


@app.route("/ip/confirm_save", methods=['POST'])
def confirm_save_endpoint():
    """结果确认保存接口"""
    try:
        data = request.get_json()
        result_id = data.get('result_id')
        save = data.get('save')

        if result_id is None or save is None:
            print(f"[ERROR] 参数缺失 - result_id: {result_id}, save: {save}")
            return jsonify({"code": 400, "message": "缺少必要参数"}), 400

        if save not in [0, 1]:
            print(f"[ERROR] 无效save值 - result_id: {result_id}, save: {save}")
            return jsonify({"code": 400, "message": "save参数值无效（必须为0或1）"}), 400

        if save == 1:
            print(f"[INFO] 数据保留成功 - result_id: {result_id}")
            return jsonify({"code": 200, "message": "数据已保留"}), 200

        result = con_mysql("SELECT image_path FROM result WHERE result_id = %s", (result_id,))
        if not result:
            print(f"[ERROR] 记录不存在 - result_id: {result_id}")
            return jsonify({"code": 404, "message": f"结果记录未找到: {result_id}"}), 404

        image_path = result['image_path']

        try:
            con_mysql("DELETE FROM result WHERE result_id = %s", (result_id,))
            print(f"[INFO] 数据库记录删除成功 - result_id: {result_id}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"[INFO] 文件删除成功 - path: {image_path}")
            else:
                print(f"[WARNING] 文件未找到 - path: {image_path}")
            return jsonify({"code": 200, "message": "数据已成功删除"}), 200
        
        except Exception as e:
            print(f"[ERROR] 数据删除失败 - result_id: {result_id}, 错误: {str(e)}")
            return jsonify({"code": 500, "message": "数据删除失败"}), 500

    except Exception as e:
        print(f"[ERROR] 确认保存接口异常 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": "服务器内部错误"}), 500


@app.route("/ip/get_result_image", methods=['GET', 'POST'])
def get_result_image():
    """根据result_id返回对应的处理结果图片"""
    try:
        if request.method == 'GET':
            result_id = request.args.get('result_id')
        else:  # POST
            data = request.get_json()
            result_id = data.get('result_id') if data else None

        if not result_id:
            print(f"[ERROR] 缺少必要参数 - result_id: {result_id}")
            return jsonify({"code": 400, "message": "缺少必要参数result_id"}), 400

        result = con_mysql("SELECT image_path FROM result WHERE result_id = %s", (result_id,))
        if not result:
            print(f"[ERROR] 记录不存在 - result_id: {result_id}")
            return jsonify({"code": 404, "message": f"结果记录未找到: {result_id}"}), 404

        image_path = result['image_path']

        # 验证路径是否符合硬编码目录（防止路径篡改）
        if not image_path.startswith(UPLOAD_FOLDER):
            print(f"[ERROR] 非法路径访问: {image_path}")
            return jsonify({"code": 403, "message": "禁止访问"}), 403

        if not os.path.exists(image_path):
            print(f"[ERROR] 图片文件不存在 - path: {image_path}")
            return jsonify({"code": 404, "message": "图片文件不存在"}), 404

        print(f"[INFO] 成功返回图片 - result_id: {result_id}, path: {image_path}")
        return send_file(image_path, mimetype='image/jpeg')

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库查询失败 - result_id: {result_id}, 错误: {str(e)}")
        return jsonify({"code": 500, "message": f"数据库操作失败: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] 获取结果图片接口异常 - 错误: {str(e)}")
        return jsonify({"code": 500, "message": "服务器内部错误，请稍后重试"}), 500


@app.route("/ip/get_user_results", methods=['GET', 'POST'])
def get_user_result():
    """根据用户名查询结果表中的所有结果"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
        else:  # POST
            data = request.get_json()
            user_id = data.get('user_id')

        user = User.get_by_user_id(user_id)
        if not user:
            print(f"[ERROR] 用户未找到: {user_id}")
            return jsonify({"code": 404, "message": f"User not found: {user_id}"}), 404

        team_name = get_team_name_by_user_id(user_id)

        team = Team.get_all_by_name(team_name)
        area_bound = team.area_bound if team else None
        
        query = """
        SELECT result_id, gps_location, detection_time, defect_type, severity 
        FROM result 
        WHERE user_id = %s
        """
        results = con_mysql(query, (user_id,), fetch_all=True)

        result_list = []
        for row in results:
            result_dict = {
                'result_id': row['result_id'],
                'gps_location': row['gps_location'],
                'detection_time': row['detection_time'],
                'defect_type': row['defect_type'],
                'severity': row['severity']
            }
            result_list.append(result_dict)

        print(f"[INFO] 成功查询到{len(result_list)}条结果，用户: {user.username}")
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": result_list,
            "boundary": area_bound
        })

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库操作失败: {str(e)}")
        return jsonify({"code": 500, "message": f"Database operation failed: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 获取用户结果异常: {str(e)}")
        return jsonify({"code": 500, "message": "Internal server error"}), 500


@app.route("/ip/get_team_results", methods=['GET', 'POST'])
def get_team_results():
    """根据用户名查询结果表中的所有结果"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
        else:  # POST
            data = request.get_json()
            user_id = data.get('user_id')

        team_name = get_team_name_by_user_id(user_id)

        team = Team.get_all_by_name(team_name)
        if not team:
            print(f"[WARNING] 小组未找到: {team_name}")
            return jsonify({"code": 404, "message": f"Team not found: {team_name}"}), 404
        
        user_ids = team.get_all_members()

        # 修改查询语句：添加 LEFT JOIN 获取 username
        query = """
        SELECT 
            r.result_id, 
            r.gps_location, 
            r.detection_time, 
            r.defect_type, 
            r.severity,
            u.username  -- 添加 username 字段
        FROM result r
        LEFT JOIN user u ON r.user_id = u.user_id  -- 关联 user 表
        WHERE r.user_id IN %s
        """
        results = con_mysql(query, (user_ids,), fetch_all=True)

        result_list = []
        for row in results:
            result_dict = {
                'result_id': row['result_id'],
                'username': row['username'],
                'gps_location': row['gps_location'],
                'detection_time': row['detection_time'],
                'defect_type': row['defect_type'],
                'severity': row['severity']
            }
            result_list.append(result_dict)

        print(f"[INFO] 成功查询到{len(result_list)}条结果，小组: {team_name}")
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": result_list,
            "boundary": team.area_bound
        })

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库操作失败: {str(e)}")
        return jsonify({"code": 500, "message": f"Database operation failed: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 获取小组结果异常: {str(e)}")
        return jsonify({"code": 500, "message": "Internal server error"}), 500


@app.route("/ip/get_user_results_time", methods=['POST'])
def get_user_results_time():
    """根据用户名查询结果表中的所有结果"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        start = data.get('start_time')
        end = data.get('end_time')

        user = User.get_by_user_id(user_id)
        if not user:
            print(f"[ERROR] 用户未找到: {user_id}")
            return jsonify({"code": 404, "message": f"User not found: {user_id}"}), 404

        team_name = get_team_name_by_user_id(user_id)

        team = Team.get_all_by_name(team_name)
        area_bound = team.area_bound if team else None
        
        query = """
        SELECT result_id, gps_location, detection_time, defect_type, severity 
        FROM result 
        WHERE user_id = %s AND detection_time BETWEEN %s AND %s
        """
        results = con_mysql(query, (user_id, start, end), fetch_all=True)

        result_list = []
        for row in results:
            result_dict = {
                'result_id': row['result_id'],
                'gps_location': row['gps_location'],
                'detection_time': row['detection_time'],
                'defect_type': row['defect_type'],
                'severity': row['severity']
            }
            result_list.append(result_dict)

        print(f"[INFO] 成功查询到{len(result_list)}条结果，用户: {user.username}")
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": result_list,
            "boundary": area_bound
        })

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库操作失败: {str(e)}")
        return jsonify({"code": 500, "message": f"Database operation failed: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 获取用户结果异常: {str(e)}")
        return jsonify({"code": 500, "message": "Internal server error"}), 500   
    

@app.route("/ip/get_team_results_time", methods=['POST'])
def get_team_results_time():
    """根据用户名查询结果表中的所有结果"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        start = data.get('start_time')
        end = data.get('end_time')

        team_name = get_team_name_by_user_id(user_id)

        team = Team.get_all_by_name(team_name)
        if not team:
            print(f"[WARNING] 小组未找到: {team_name}")
            return jsonify({"code": 404, "message": f"Team not found: {team_name}"}), 404
        
        user_ids = team.get_all_members()

        # 修改查询语句：添加 LEFT JOIN 获取 username
        query = """
        SELECT 
            r.result_id, 
            r.gps_location, 
            r.detection_time, 
            r.defect_type, 
            r.severity,
            u.username  -- 添加 username 字段
        FROM result r
        LEFT JOIN user u ON r.user_id = u.user_id  -- 关联 user 表
        WHERE r.user_id IN %s AND r.detection_time BETWEEN %s AND %s
        """
        results = con_mysql(query, (user_ids, start, end), fetch_all=True)

        result_list = []
        for row in results:
            result_dict = {
                'result_id': row['result_id'],
                'username': row['username'],
                'gps_location': row['gps_location'],
                'detection_time': row['detection_time'],
                'defect_type': row['defect_type'],
                'severity': row['severity']
            }
            result_list.append(result_dict)

        print(f"[INFO] 成功查询到{len(result_list)}条结果，小组: {team_name}")
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": result_list,
            "boundary": team.area_bound
        })

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库操作失败: {str(e)}")
        return jsonify({"code": 500, "message": f"Database operation failed: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 获取小组结果异常: {str(e)}")
        return jsonify({"code": 500, "message": "Internal server error"}), 500
