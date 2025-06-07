# import pymysql
# from dbutils.pooled_db import PooledDB
# import threading

# # 数据库连接池配置
# pool = PooledDB(
#     creator=pymysql,  # 使用pymysql作为数据库驱动
#     host="localhost",
#     port=3306,
#     user="root",
#     password="123456",
#     database="rdds",
#     charset="utf8mb4",
#     mincached=5,      # 连接池中空闲连接的初始数量
#     maxcached=15,      # 连接池中空闲连接的最大数量
#     maxshared=5,      # 共享连接的最大数量
#     maxconnections=30, # 连接池允许的最大连接数
#     blocking=True,    # 连接池达到最大连接数时是否阻塞
#     connect_timeout=15,  # 连接超时时间（秒）
#     read_timeout=30,     # 读取超时时间（秒）
#     write_timeout=30,    # 写入超时时间（秒）
# )

# # 线程局部存储，用于保存每个线程的数据库连接
# thread_local = threading.local()

# def get_connection() -> pymysql.connections.Connection:
#     """获取当前线程的数据库连接
    
#     每个线程使用独立的数据库连接，避免线程间共享连接导致的问题
    
#     Returns:
#         pymysql.connections.Connection: 数据库连接对象
#     """
#     # 检查当前线程是否已有连接
#     if not hasattr(thread_local, 'connection'):
#         thread_local.connection = pool.connection()
#     return thread_local.connection

# def close_connection() -> None:
#     """关闭当前线程的数据库连接"""
#     if hasattr(thread_local, 'connection'):
#         thread_local.connection.close()
#         del thread_local.connection

# def con_mysql(sql_code: str, params: tuple = None, fetch_all: bool = False) -> any:
#     """执行SQL语句并处理结果（线程安全版本）

#     Args:
#         sql_code (str): SQL语句
#         params (tuple, optional): SQL参数，用于预处理语句. 默认None.
#         fetch_all (bool, optional): 是否返回全部查询结果. 
#                                     仅对SELECT语句有效. 默认False.

#     Returns:
#         any: 
#             - SELECT语句: 返回字典形式的查询结果（单条或多条）
#             - 非SELECT语句: 返回受影响的行数

#     Raises:
#         pymysql.MySQLError: 数据库操作失败时抛出异常

#     Notes:
#         1. 使用数据库连接池管理连接
#         2. 每个线程使用独立的数据库连接
#         3. 使用字典游标返回结果（DictCursor）
#         4. 对于SELECT语句，根据fetch_all决定返回单条/多条记录
#         5. 对于非SELECT语句，自动提交事务
#         6. 发生异常时自动回滚事务
#     """
#     conn = get_connection()
#     try:
#         # 使用字典游标，结果以字典形式返回
#         with conn.cursor(pymysql.cursors.DictCursor) as cursor:
#             cursor.execute(sql_code, params)
            
#             # 查询语句处理逻辑
#             if sql_code.strip().lower().startswith("select"):
#                 return cursor.fetchall() if fetch_all else cursor.fetchone()
            
#             # 非查询语句（INSERT/UPDATE/DELETE等）处理逻辑
#             else:
#                 conn.commit()  # 提交事务
#                 return cursor  # 返回受影响的行数
                
#     except pymysql.MySQLError as e:
#         conn.rollback()  # 异常时回滚事务
#         raise e  # 向上抛出异常
#     finally:
#         # 可以选择在此处关闭连接，或者根据需要保持连接
#         pass

# import pymysql
 
# conn = pymysql.connect(host="localhost", port=3306,
#                        user="root", password="123456",
#                        database="rdds", charset="utf8mb4")
 

# def con_mysql(sql_code, params=None, fetch_all=False):
#     try:
#         conn.ping(reconnect=True)
#         with conn.cursor(pymysql.cursors.DictCursor) as cursor:
#             cursor.execute(sql_code, params)
            
#             if sql_code.strip().lower().startswith("select"):
#                 # 对于查询语句，返回结果集
#                 return cursor.fetchall() if fetch_all else cursor.fetchone()
#             else:
#                 conn.commit()
#                 return cursor
#     except pymysql.MySQLError as e:
#         conn.rollback()
#         raise e
import pymysql
from pymysql import OperationalError
import threading

# 使用线程局部存储管理每个线程的数据库连接
local = threading.local()

def get_db_connection():
    """获取线程安全的数据库连接"""
    if not hasattr(local, 'connection') or local.connection is None:
        try:
            # 创建新连接
            local.connection = pymysql.connect(
                host="localhost", 
                port=3306,
                user="root", 
                password="123456",
                database="rdds", 
                charset="utf8mb4",
                autocommit=True  # 自动提交事务
            )
        except Exception as e:
            print(f"创建数据库连接失败: {e}")
            local.connection = None
    return local.connection

def con_mysql(sql_code, params=None, fetch_all=False, retries=3):
    """执行SQL语句，支持自动重试"""
    attempt = 0
    while attempt < retries:
        try:
            # 获取当前线程的连接
            conn = get_db_connection()
            
            # 检查连接有效性
            if conn is None or not conn.open:
                conn = get_db_connection()  # 重新创建连接
            
            # 使用上下文管理器确保资源释放
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql_code, params)
                
                if sql_code.strip().lower().startswith("select"):
                    # 对于查询语句，返回结果集
                    return cursor.fetchall() if fetch_all else cursor.fetchone()
                else:
                    conn.commit()
                    return cursor
                    
        except OperationalError as e:
            # 处理连接断开的情况
            conn.rollback()
            if e.args[0] in (2006, 2013, 2014, 2045, 2055):  # MySQL断开连接错误码
                print(f"数据库连接丢失，正在重试 ({attempt+1}/{retries})")
                # 重置连接
                if hasattr(local, 'connection'):
                    local.connection = None
                attempt += 1
            else:
                raise
        except Exception as e:
            print(f"数据库操作失败: {e}")
            raise
    raise OperationalError("达到最大重试次数，无法执行查询")
