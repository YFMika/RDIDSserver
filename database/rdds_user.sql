-- MySQL dump 10.13  Distrib 8.0.42, for Win64 (x86_64)
--
-- Host: localhost    Database: rdds
-- ------------------------------------------------------
-- Server version	8.0.42

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(100) NOT NULL COMMENT '密码',
  `user_type` tinyint NOT NULL COMMENT '用户类型：0=作业人员，1=管理员',
  `name` varchar(20) DEFAULT NULL COMMENT '姓名',
  `gender` enum('男','女') DEFAULT NULL COMMENT '性别',
  `phone` varchar(11) DEFAULT NULL COMMENT '手机号',
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `username_2` (`username`),
  UNIQUE KEY `phone` (`phone`),
  KEY `idx_user_type` (`user_type`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户信息表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
INSERT INTO `user` VALUES (1,'admin','123456',1,'姚飞扬','男','13800138001'),(2,'worker1','123456',0,'张三','男','13900139001'),(3,'worker2','123456',0,'李四','女','13700137001'),(4,'worker3','123456',0,'王五','男','13600136001'),(5,'worker4','123456',0,'赵六','女','13500135001'),(6,'yfy','12345678',1,'云枫','男','13812345678'),(7,'fadassadasd','$2b$12$Hnxw8nMV12yTSCc8oe0gA.rf93qxBrClnkYYN6SHMtPde1V5HbtdC',0,NULL,NULL,NULL),(8,'worker10','$2b$12$2AGQF/0jogYCGXKMUuOIme1oU.ZjqY4Wf0hS2RIxANjzeB2a7v6ty',0,NULL,NULL,NULL),(10,'wxh123','$2b$12$0Yx8o9yVkLyFEjdbPBG4Oux2qCrTjhkXAv5gRGu2TzDePpCG37KtC',0,NULL,NULL,NULL),(11,'hyperheer2','$2b$12$snqql.8TA6xJPN27reueXuIi4JadYhUVsUGr5TnoYq2hgRm6eLmJS',1,'王可岩','女','18538078570'),(12,'hwjfhdbbn','$2b$12$MzVc/LHCchtJFFiMgpZ57OvKwYMS0YDX5cTgWkdBuALjm2ftJJ9qm',0,NULL,NULL,NULL),(13,'hyperheer','$2b$12$1I2xhL..GoJagDPY95v2L.HolFdGhNVloG0xE29168stw3xVfxOJ6',0,'王可岩','男','18538078579'),(14,'jwhdhb','$2b$12$BEilsag6bmTNppi/SSJcMu/6bGuA2jJQDnnwxR1RpF.AxjLpGQ8L6',0,NULL,NULL,NULL),(15,'jwhdhjdj','$2b$12$MjEVqFVycxf5XsixciTJceRQsCiQwtiNiOqimUW3pBDo4RboH6CPS',0,NULL,NULL,NULL),(16,'yfy123','$2b$12$7qMl/0wzHj96IulKEVNcIuUoUwqlN.Ns9X1HMewgf5SFYPa1gc3ca',1,'姚飞扬','女','15756002133');
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-31 21:51:37
