# import pymysql
# from pymysql.cursors import DictCursor
# from src.ConnectDB import con_mysql

# class Area:
#     def __init__(self, area_id=None, team_id=None, user_id=None, gps_data=None, status='未分配'):
#         self.area_id = area_id
#         self.team_id = team_id
#         self.user_id = user_id
#         self.gps_data = gps_data  # GEOMETRY类型建议存储WKT格式字符串
#         self.status = status

#     @staticmethod
#     def get_by_id(area_id):
#         """通过区域编号查询区域"""
#         sql = "SELECT * FROM area WHERE area_id = %s"
#         result = con_mysql(sql, (area_id,))
#         return Area(**result) if result else None

#     @staticmethod
#     def get_by_team(team_id):
#         """获取小组所有区域"""
#         sql = "SELECT * FROM area WHERE team_id = %s"
#         results = con_mysql(sql, (team_id,), fetch_all=True)
#         return [Area(**row) for row in results] if results else []

#     @staticmethod
#     def get_unassigned_areas():
#         """获取所有未分配区域"""
#         sql = "SELECT * FROM area WHERE status = '未分配'"
#         results = con_mysql(sql, fetch_all=True)
#         return [Area(**row) for row in results] if results else []

#     def save(self):
#         """保存区域（新增或更新）"""
#         if self.area_id:
#             # 更新区域
#             sql = """
#             UPDATE area SET 
#                 team_id = %s, 
#                 user_id = %s, 
#                 gps_data = ST_GeomFromText(%s), 
#                 status = %s
#             WHERE area_id = %s
#             """
#             params = (self.team_id, self.user_id, self.gps_data, self.status, self.area_id)
#         else:
#             # 新增区域
#             sql = """
#             INSERT INTO area (area_id, team_id, user_id, gps_data, status)
#             VALUES (%s, %s, %s, ST_GeomFromText(%s), %s)
#             """
#             params = (self.area_id, self.team_id, self.user_id, self.gps_data, self.status)
        
#         cursor = con_mysql(sql, params)
#         return cursor is not None

#     def delete(self):
#         """删除区域"""
#         if not self.area_id:
#             return False
        
#         sql = "DELETE FROM area WHERE area_id = %s"
#         cursor = con_mysql(sql, (self.area_id,))
#         return cursor and cursor.rowcount > 0

#     def to_dict(self):
#         """转换为字典格式（GEOMETRY转WKT）"""
#         return {
#             'area_id': self.area_id,
#             'team_id': self.team_id,
#             'user_id': self.user_id,
#             'gps_data': self.gps_data,
#             'status': self.status
#         }
    