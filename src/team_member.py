import pymysql
from pymysql.cursors import DictCursor
from src.ConnectDB import con_mysql

class Team_member:
    def __init__(self, team_id=None, user_id=None):
        self.team_id = team_id
        self.user_id = user_id

    @staticmethod
    def get_by_team_and_user(team_id, user_id):
        """通过小组ID和用户ID查询组员"""
        sql = "SELECT * FROM team_member WHERE team_id = %s AND user_id = %s"
        result = con_mysql(sql, (team_id, user_id))
        return Team_member(**result) if result else None

    @staticmethod
    def get_members_by_team(team_id):
        """获取小组所有组员"""
        sql = "SELECT user_id FROM team_member WHERE team_id = %s"
        results = con_mysql(sql, (team_id,), fetch_all=True)
        return [row['user_id'] for row in results] if results else []

    @staticmethod
    def get_teams_by_user(user_id):
        """获取用户所属的所有小组"""
        sql = "SELECT team_id FROM team_member WHERE user_id = %s"
        results = con_mysql(sql, (user_id,), fetch_all=True)
        return [row['team_id'] for row in results] if results else []
    
    @staticmethod
    def get_teams_name_by_user(user_id):
        """获取用户所属的所有小组名字"""
        # 使用JOIN连接team_member和Team表，通过team_id关联
        sql = """
        SELECT t.team_name 
        FROM team_member tm
        JOIN Team t ON tm.team_id = t.team_id
        WHERE tm.user_id = %s
        """
        results = con_mysql(sql, (user_id,), fetch_all=True)
        return [row['team_name'] for row in results] if results else []

    def save(self):
        """添加组员（小组ID和用户ID必须同时存在）"""
        if not self.team_id or not self.user_id:
            return False
        
        sql = "INSERT INTO team_member (team_id, user_id) VALUES (%s, %s)"
        cursor = con_mysql(sql, (self.team_id, self.user_id))
        return cursor is not None

    def delete(self):
        """移除组员"""
        if not self.team_id or not self.user_id:
            return False
        
        sql = "DELETE FROM team_member WHERE team_id = %s AND user_id = %s"
        cursor = con_mysql(sql, (self.team_id, self.user_id))
        return cursor and cursor.rowcount > 0
    
    def disband(self):
        """解散小组"""
        if not self.team_id or not self.user_id:
            return False
        
        sql = "DELETE FROM team_member WHERE team_id = %s"
        cursor1 = con_mysql(sql, (self.team_id,))

        sql = "DELETE FROM team WHERE team_id = %s"
        cursor2 = con_mysql(sql, (self.team_id,))
        return cursor1 and cursor1.rowcount > 0 and cursor2 and cursor2.rowcount > 0