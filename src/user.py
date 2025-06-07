import pymysql
from pymysql.cursors import DictCursor
from src.ConnectDB import con_mysql

class User:
    def __init__(self, user_id=None, username=None, password=None, 
                 user_type=0, name=None, gender=None, phone=None):
        self.user_id = user_id
        self.username = username
        self.password = password
        self.user_type = user_type
        self.name = name
        self.gender = gender
        self.phone = phone

    @staticmethod
    def get_by_username(username):
        """通过用户名查询用户"""
        sql = "SELECT * FROM user WHERE username = %s"
        result = con_mysql(sql, (username,))  # 直接获取查询结果（字典或None）
        return User(**result) if result else None  # 如果结果不为空，实例化User对象
    
    @staticmethod
    def get_by_user_id(user_id):
        """通过用户名查询用户"""
        sql = "SELECT * FROM user WHERE user_id = %s"
        result = con_mysql(sql, (user_id,))  # 直接获取查询结果（字典或None）
        return User(**result) if result else None  # 如果结果不为空，实例化User对象
    
    @staticmethod
    def whether_user_in_team(user_id):
        """查询用户是否在某个小组中"""
        sql = "SELECT * FROM team_member WHERE user_id = %s"
        result = con_mysql(sql, (user_id,))  # 直接获取查询结果（字典或None）
        return True if result else False

    def save(self):
        """保存用户（新增或更新）"""
        if self.user_id:
            # 更新用户
            sql = """
            UPDATE user SET 
                username = %s, password = %s, 
                user_type = %s, name = %s, 
                gender = %s, phone = %s 
            WHERE user_id = %s
            """
            params = (
                self.username, self.password, 
                self.user_type, self.name, 
                self.gender, self.phone, 
                self.user_id
            )
        else:
            # 新增用户
            sql = """
            INSERT INTO user 
                (username, password, user_type, name, gender, phone) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                self.username, self.password, 
                self.user_type, self.name, 
                self.gender, self.phone
            )
        
        cursor = con_mysql(sql, params)
        if cursor and not self.user_id:
            self.user_id = cursor.lastrowid
        return self.user_id is not None

    def delete(self):
        """删除用户"""
        if not self.user_id:
            return False
        
        sql = "DELETE FROM user WHERE user_id = %s"
        cursor = con_mysql(sql, (self.user_id,))
        return cursor and cursor.rowcount > 0

    def to_dict(self):
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'user_type': self.user_type,
            'name': self.name,
            'gender': self.gender,
            'phone': self.phone,
        }
