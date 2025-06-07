import pymysql
from pymysql.cursors import DictCursor
from src.ConnectDB import con_mysql

class Result:
    def __init__(self, result_id=None, user_id=None, area_name=None, image_path=None, 
                 gps_location=None, detection_time=None, defect_type=None, severity=0.0):
        self.result_id = result_id
        self.user_id = user_id
        self.area_name = area_name
        self.image_path = image_path
        self.gps_location = gps_location  # 存储为'经度,纬度'格式
        self.detection_time = detection_time  # datetime类型需转换为字符串（如'2023-10-01 12:00:00'）
        self.defect_type = defect_type
        self.severity = severity

    @staticmethod
    def get_by_id(result_id):
        """通过结果ID查询检测结果"""
        sql = "SELECT * FROM detection_result WHERE result_id = %s"
        result = con_mysql(sql, (result_id,))
        return Result(**result) if result else None

    @staticmethod
    def get_by_user(user_id):
        """获取用户所有检测结果"""
        sql = "SELECT * FROM detection_result WHERE user_id = %s"
        results = con_mysql(sql, (user_id,), fetch_all=True)
        return [Result(**row) for row in results] if results else []

    @staticmethod
    def get_latest_results(limit=10):
        """获取最新检测结果（按时间倒序）"""
        sql = "SELECT * FROM detection_result ORDER BY detection_time DESC LIMIT %s"
        results = con_mysql(sql, (limit,), fetch_all=True)
        return [Result(**row) for row in results] if results else []

    def save(self):
        """保存检测结果（新增或更新，result_id为AUTO_INCREMENT，新增时无需传参）"""
        if self.result_id:
            # 更新结果（通常用于修改缺陷类型或精度）
            sql = """
            UPDATE detection_result SET 
                user_id = %s, 
                area_name = %s, 
                image_path = %s, 
                gps_location = %s, 
                detection_time = STR_TO_DATE(%s, %%Y-%%m-%%d %%H:%%i:%%s), 
                defect_type = %s, 
                severity = %s
            WHERE result_id = %s
            """
            params = (self.user_id, self.area_name, self.image_path, self.gps_location,
                      self.detection_time, self.defect_type, self.severity, self.result_id)
        else:
            # 新增结果
            sql = """
            INSERT INTO detection_result (user_id, area_name, image_path, gps_location, 
                                          detection_time, defect_type, accuracy)
            VALUES (%s, %s, %s, %s, STR_TO_DATE(%s, %%Y-%%m-%%d %%H:%%i:%%s), %s, %s)
            """
            params = (self.user_id, self.area_name, self.image_path, self.gps_location,
                      self.detection_time, self.defect_type, self.severity)
        
        cursor = con_mysql(sql, params)
        if cursor and not self.result_id:
            self.result_id = cursor.lastrowid  # 获取自增ID
        return cursor is not None

    def delete(self):
        """删除检测结果"""
        if not self.result_id:
            return False
        
        sql = "DELETE FROM detection_result WHERE result_id = %s"
        cursor = con_mysql(sql, (self.result_id,))
        return cursor and cursor.rowcount > 0

    def to_dict(self):
        """转换为字典格式（datetime转字符串，精度保留两位小数）"""
        return {
            'result_id': self.result_id,
            'user_id': self.user_id,
            'area_name': self.area_name,
            'image_path': self.image_path,
            'gps_location': self.gps_location,
            'detection_time': self.detection_time.strftime('%Y-%m-%d %H:%M:%S') if self.detection_time else None,
            'defect_type': self.defect_type,
            'severity': round(self.severity, 2)
        }