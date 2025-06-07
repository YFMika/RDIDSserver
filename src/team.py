import pymysql
from pymysql.cursors import DictCursor
from src.ConnectDB import con_mysql

class Team:
    def __init__(self, team_id=None, team_name=None, leader_id=None, task_area=None, area_bound=None):
        self.team_id = team_id
        self.team_name = team_name
        self.leader_id = leader_id
        self.task_area = task_area
        self.area_bound = area_bound

    @staticmethod
    def get_by_id(team_id):
        """通过ID查询小组"""
        sql = """
        SELECT team_id, 
            team_name, 
            leader_id, 
            task_area
        FROM team
        WHERE team_id = %s
        """
        result = con_mysql(sql, (team_id,))
        return Team(**result) if result else None

    @staticmethod
    def get_by_name(team_name):
        """通过名称查询小组"""
        sql = """
        SELECT team_id, 
            team_name, 
            leader_id, 
            task_area
        FROM team
        WHERE team_name = %s
        """
        result = con_mysql(sql, (team_name,))
        return Team(**result) if result else None
    
    @staticmethod
    def get_all_by_name(team_name):
        """通过名称查询小组"""
        sql = "SELECT * FROM team WHERE team_name = %s"
        result = con_mysql(sql, (team_name,))
        return Team(**result) if result else None
    
    @staticmethod
    def get_boundary(team_name):
        """通过名称查询边界"""
        sql = """
        SELECT task_area,
            area_bound
        FROM team
        WHERE team_name = %s
        """
        result = con_mysql(sql, (team_name,))
        return Team(**result) if result else None

    @staticmethod
    def get_all():
        """获取所有小组"""
        sql = "SELECT * FROM team"
        results = con_mysql(sql, fetch_all=True)
        return [Team(**row) for row in results] if results else []
    
    def get_all_members(self):
        """获取小组中所有成员id"""
        sql = "SELECT user_id FROM team_member WHERE team_id = %s"
        results = con_mysql(sql, (self.team_id,), fetch_all=True)
        return [result['user_id'] for result in results] if results else []

    def save(self):
        """保存小组（新增或更新）"""
        if self.team_id:
            # 更新小组
            sql = """
            UPDATE team SET 
                team_name = %s, 
                leader_id = %s, 
                task_area = %s,
                area_bound = %s
            WHERE team_id = %s
            """
            params = (self.team_name, self.leader_id, self.task_area, self.area_bound, self.team_id)
        else:
            # 新增小组（team_id需提前生成，如UUID）
            sql = """
            INSERT INTO team (team_name, leader_id, task_area, area_bound)
            VALUES (%s, %s, %s, %s)
            """
            params = (self.team_name, self.leader_id, self.task_area, self.area_bound)
        
        cursor = con_mysql(sql, params)
        return cursor is not None

    def delete(self):
        """删除小组"""
        if not self.team_id:
            return False
        
        sql = "DELETE FROM team WHERE team_id = %s"
        cursor = con_mysql(sql, (self.team_id,))
        return cursor and cursor.rowcount > 0

    def to_dict(self):
        """转换为字典格式（GEOMETRY转WKT）"""
        return {
            'team_id': self.team_id,
            'team_name': self.team_name,
            'leader_id': self.leader_id,
            'task_area': self.task_area  # 建议返回ST_AsText(task_area)的结果
        }