from flask import Flask, request, jsonify
from src.ConnectDB import con_mysql  
import pymysql
from src.team import Team

app = Flask(__name__)

# @app.route("/ad/area_division", methods=["POST"])
# def area_division():
#     """
#     区域分配接口：批量添加或更新区域并分配给指定用户

#     接收包含多个区域信息的JSON数组，每个区域信息需包含area_id、team_name、gps_data和user_id。
#     系统会先检查数据库中是否已有该区域记录（通过area_id判断）：
#     - 若无记录，则插入新数据并将状态设置为"已分配"
#     - 若已有记录，则更新该记录的team_id、gps_data、user_id和status

#     Args:
#         request.json (list): 区域信息列表，每个区域为包含以下字段的字典
#             - area_id (str): 区域唯一标识
#             - team_name (int): 所属小组名字
#             - gps_data (str): 区域GPS坐标数据
#             - user_id (int): 分配的用户ID

#     Returns:
#         flask.Response: JSON响应
#             - 成功时: {"code": 200, "message": "成功处理X条区域数据，插入X条，更新X条"}
#             - 失败时: {"code": 错误码, "message": "错误信息"}

#     Raises:
#         400 Bad Request: 请求格式错误（非数组、空数组或缺少必要字段）
#         500 Internal Server Error: 服务器处理失败

#     Examples:
#         请求示例:
#         [
#             {
#                 "area_id": "A001",
#                 "team_name": "yfy",
#                 "gps_data": "30.1234,120.5678",
#                 "user_id": 1001
#             }
#         ]

#         响应示例:
#         {
#             "code": 200,
#             "message": "成功处理1条区域数据，插入0条，更新1条"
#         }
#     """
#     try:
#         # 解析客户端请求（保持不变）
#         area_list = request.json
#         if not isinstance(area_list, list) or len(area_list) == 0:
#             print(f"[WARNING] json为空")
#             return jsonify({
#                 "code": 400,
#                 "message": "请求数据必须为非空的JSON数组"
#             }), 400

#         # 校验字段（保持不变）
#         required_fields = ["area_id", "team_name", "gps_data", "user_id"]
#         for item in area_list:
#             if not all(key in item for key in required_fields):
#                 missing_fields = [key for key in required_fields if key not in item]
#                 print(f"[WARNING] json缺少必要字段")
#                 return jsonify({
#                     "code": 400,
#                     "message": f"缺少必要字段: {', '.join(missing_fields)}"
#                 }), 400

#         insert_count = 0
#         update_count = 0
#         failed_items = []

#         # 批量处理每个区域
#         for item in area_list:
#             try:
#                 # 获取team_id（保持不变）
#                 team = Team.get_by_name(item['team_name'])
#                 if not team:
#                     raise ValueError(f"小组不存在: {item['team_name']}")
#                 team_id = team.team_id
                
#                 # 检查区域是否已存在
#                 check_sql = "SELECT area_id FROM area WHERE area_id = %s"
#                 existing_record = con_mysql(check_sql, (item['area_id'],), fetch_all=False)
                
#                 if existing_record:
#                     # 存在记录，执行更新
#                     update_sql = """
#                     UPDATE area 
#                     SET team_id = %s, 
#                         gps_data = %s, 
#                         user_id = %s, 
#                         status = '已分配',
#                         update_time = NOW()  -- 可选：添加更新时间戳
#                     WHERE area_id = %s
#                     """
#                     params = (
#                         team_id,
#                         item['gps_data'],
#                         item['user_id'],
#                         item['area_id']
#                     )
#                     cursor = con_mysql(update_sql, params)
#                     if cursor.rowcount >= 1:
#                         update_count += 1
#                     else:
#                         raise ValueError(f"更新失败，区域ID: {item['area_id']}")
#                 else:
#                     # 不存在记录，执行插入
#                     insert_sql = """
#                     INSERT INTO area 
#                     (area_id, team_id, gps_data, user_id, status, create_time) 
#                     VALUES (%s, %s, %s, %s, '已分配', NOW())  -- 可选：添加创建时间戳
#                     """
#                     params = (
#                         item['area_id'],
#                         team_id,
#                         item['gps_data'],
#                         item['user_id']
#                     )
#                     cursor = con_mysql(insert_sql, params)
#                     if cursor.rowcount == 1:
#                         insert_count += 1
#                     else:
#                         raise ValueError(f"插入失败，区域ID: {item['area_id']}")
            
#             except pymysql.MySQLError as e:
#                 # 处理数据库错误
#                 failed_items.append({
#                     "item": item,
#                     "error": f"数据库错误: {str(e)}"
#                 })
#                 continue
#             except Exception as e:
#                 failed_items.append({
#                     "item": item,
#                     "error": f"处理错误: {str(e)}"
#                 })
#                 continue

#         # 构造响应
#         total_success = insert_count + update_count
#         response_data = {
#             "code": 200,
#             "message": f"成功处理 {total_success} 条区域数据，插入 {insert_count} 条，更新 {update_count} 条"
#         }
#         print(f"[INFO] 处理完成: 插入 {insert_count} 条，更新 {update_count} 条")
        
#         if failed_items:
#             response_data["failed_items"] = failed_items
#             print(f"[WARNING] 部分记录处理失败: {len(failed_items)} 条")
        
#         return jsonify(response_data)

#     except Exception as e:
#         print(f"[ERROR] 服务器错误: {str(e)}")
#         return jsonify({
#             "code": 500,
#             "message": f"服务器处理失败: {str(e)}"
#         }), 500
    

@app.route("/ad/get_team_area", methods=['GET', 'POST'])
def get_team_area():
    """根据用户名查询结果表中的所有结果"""
    try:
        if request.method == 'GET':
            team_name = request.args.get('team_name')
        else:  # POST
            data = request.get_json()
            team_name = data.get('team_name')

        team = Team.get_by_name(team_name)
        if not team:
            print(f"[ERROR] 小组未找到: {team_name}")
            return jsonify({"code": 404, "message": f"Team not found: {team_name}"}), 404

        query = """
        SELECT area_name 
        FROM team
        WHERE team_name = %s
        """
        area_name = con_mysql(query, (team_name,))

        print(f"[INFO] 成功查询到小组 {team_name} 的区域 {area_name}")
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": area_name
        })

    except pymysql.MySQLError as e:
        print(f"[ERROR] 数据库操作失败: {str(e)}")
        return jsonify({"code": 500, "message": f"Database operation failed: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] 获取小组区域异常: {str(e)}")
        return jsonify({"code": 500, "message": "Internal server error"}), 500
    