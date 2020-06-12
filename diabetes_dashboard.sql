-- phpMyAdmin SQL Dump
-- version 5.0.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 13, 2020 at 12:07 AM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.2.27

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `diabetes_dashboard`
--

-- --------------------------------------------------------

--
-- Table structure for table `accounts`
--

CREATE TABLE `accounts` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `gender` varchar(50) NOT NULL,
  `weight` float NOT NULL,
  `heightt` float NOT NULL,
  `age` int(50) NOT NULL,
  `bmi` float NOT NULL,
  `preg` int(11) NOT NULL DEFAULT 0,
  `date` datetime DEFAULT current_timestamp(),
  `dev_id` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `accounts`
--

INSERT INTO `accounts` (`id`, `name`, `email`, `password`, `gender`, `weight`, `heightt`, `age`, `bmi`, `preg`, `date`, `dev_id`) VALUES
(1, 'test', 'test@yahoo.com', 'test', 'Male', 55, 1.675, 21, 19.6035, 0, '2020-06-02 13:10:39', 'product1'),
(2, 'atest', 'atest@yahoo.com', 'atest', 'Female', 50, 1.56, 21, 20.5457, 0, '2020-06-03 20:53:21', 'product2');

-- --------------------------------------------------------

--
-- Table structure for table `diagnoses_ml`
--

CREATE TABLE `diagnoses_ml` (
  `sno` int(50) NOT NULL,
  `glucose` float(6,3) NOT NULL,
  `height` float(4,3) NOT NULL,
  `weight` float(5,2) NOT NULL,
  `age` int(50) NOT NULL,
  `preg_num` int(50) NOT NULL DEFAULT 0,
  `date` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `diagnoses_ml`
--

INSERT INTO `diagnoses_ml` (`sno`, `glucose`, `height`, `weight`, `age`, `preg_num`, `date`) VALUES
(1, 150.000, 2.000, 50.00, 23, 0, '2020-04-21 23:54:58'),
(2, 150.000, 2.000, 16.00, 54, 0, NULL),
(3, 150.000, 2.000, 16.00, 54, 0, '2020-04-22 00:20:37'),
(4, 140.000, 3.000, 50.00, 23, 1, '2020-04-22 00:21:37'),
(5, 98.000, 2.000, 60.00, 50, 1, '2020-04-22 00:28:26'),
(6, 1.000, 1.000, 2.00, 43, 4, '2020-04-22 00:31:45'),
(7, 150.253, 1.000, 46.00, 58, 0, '2020-04-22 00:38:06'),
(8, 150.253, 1.000, 46.00, 58, 0, '2020-04-22 00:40:31'),
(9, 150.253, 1.005, 46.33, 58, 0, '2020-04-22 00:43:26'),
(10, 150.253, 1.005, 46.33, 58, 0, '2020-04-22 00:47:08'),
(11, 100.450, 1.011, 39.33, 47, 0, '2020-04-22 00:48:31'),
(12, 120.000, 1.430, 59.00, 43, 3, '2020-04-22 20:36:22'),
(13, 123.000, 1.000, 16.00, 2, 0, '2020-04-23 17:30:04'),
(14, 123.000, 1.000, 16.00, 2, 0, '2020-04-23 17:34:34'),
(15, 454.000, 9.999, 12.00, 23, 0, '2020-04-23 17:49:19'),
(16, 150.000, 1.680, 50.00, 56, 2, '2020-04-28 13:22:19'),
(17, 140.000, 2.600, 50.00, 89, 1, '2020-05-28 11:49:02'),
(18, 120.000, 1.300, 50.00, 23, 1, '2020-06-11 15:33:43'),
(19, 0.000, 0.000, 0.00, 0, 0, '2020-06-11 15:33:48'),
(20, 0.000, 0.000, 0.00, 0, 0, '2020-06-11 15:33:50'),
(21, 0.000, 0.000, 0.00, 0, 0, '2020-06-11 15:33:53'),
(22, 140.000, 1.300, 50.00, 12, 1, '2020-06-11 15:35:45'),
(23, 123.000, 1.300, 50.00, 23, 1, '2020-06-11 15:37:11');

-- --------------------------------------------------------

--
-- Table structure for table `glucose_value`
--

CREATE TABLE `glucose_value` (
  `id` int(11) NOT NULL,
  `glucose` float NOT NULL,
  `timestamp` datetime DEFAULT current_timestamp(),
  `sensor` text NOT NULL,
  `ip` text NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `glucose_value`
--

INSERT INTO `glucose_value` (`id`, `glucose`, `timestamp`, `sensor`, `ip`) VALUES
(1, -80, '2020-06-08 15:46:54', 'product1', '0'),
(2, -81, '2020-06-08 15:47:15', 'product1', '0'),
(3, -81, '2020-06-08 15:47:36', 'product1', '0'),
(4, -82, '2020-06-08 15:47:57', 'product1', '0'),
(5, -80, '2020-06-08 15:48:17', 'product1', '0'),
(6, -80, '2020-06-08 15:48:39', 'product1', '0'),
(7, -80, '2020-06-08 15:48:59', 'product1', '0'),
(8, -79, '2020-06-08 15:49:20', 'product1', '0'),
(9, -82, '2020-06-08 15:50:32', 'product1', '0'),
(10, -86, '2020-06-08 15:51:43', 'product1', '0'),
(11, -80, '2020-06-08 15:52:29', 'product1', '0'),
(12, -78, '2020-06-08 15:52:50', 'product1', '0'),
(13, -79, '2020-06-08 15:53:11', 'product1', '0'),
(14, -79, '2020-06-08 15:53:31', 'product1', '0'),
(15, -79, '2020-06-08 15:53:52', 'product1', '0'),
(16, -80, '2020-06-08 15:54:13', 'product1', '0'),
(17, -79, '2020-06-08 15:54:37', 'product1', '0'),
(18, -78, '2020-06-08 15:54:58', 'product1', '0'),
(19, -78, '2020-06-08 15:55:19', 'product1', '0'),
(20, -81, '2020-06-08 15:56:05', 'product1', '0'),
(21, -80, '2020-06-08 15:57:16', 'product1', '0'),
(22, -79, '2020-06-08 15:57:37', 'product1', '0'),
(23, -79, '2020-06-08 15:57:58', 'product1', '0'),
(24, -81, '2020-06-08 15:58:19', 'product1', '0'),
(25, -81, '2020-06-08 15:59:30', 'product1', '0'),
(26, -86, '2020-06-08 16:00:19', 'product1', '0'),
(27, -87, '2020-06-08 16:00:41', 'product1', '0'),
(28, -87, '2020-06-08 16:01:31', 'product1', '0'),
(29, -86, '2020-06-08 16:01:52', 'product1', '0'),
(30, -87, '2020-06-08 16:02:16', 'product1', '0'),
(31, -79, '2020-06-08 16:03:03', 'product1', '0'),
(32, -77, '2020-06-08 16:03:24', 'product1', '0'),
(33, -76, '2020-06-08 16:03:44', 'product1', '0'),
(34, -80, '2020-06-08 16:04:05', 'product1', '0'),
(35, -78, '2020-06-08 16:04:27', 'product1', '0'),
(36, -80, '2020-06-08 16:06:04', 'product1', '0'),
(37, -74, '2020-06-08 16:06:50', 'product1', '0'),
(38, -76, '2020-06-08 16:07:11', 'product1', '0'),
(39, -76, '2020-06-08 16:07:31', 'product1', '0'),
(40, -75, '2020-06-08 16:09:08', 'product1', '0'),
(41, -76, '2020-06-08 16:10:20', 'product1', '0'),
(42, -74, '2020-06-08 16:10:41', 'product1', '0'),
(43, -76, '2020-06-08 16:11:01', 'product1', '0'),
(44, -75, '2020-06-08 16:12:13', 'product1', '0'),
(45, -75, '2020-06-08 16:12:34', 'product1', '0'),
(46, -75, '2020-06-08 16:12:55', 'product1', '0'),
(47, -75, '2020-06-08 16:13:15', 'product1', '0'),
(48, -74, '2020-06-08 16:13:36', 'product1', '0'),
(49, -75, '2020-06-09 07:39:19', 'product1', '0'),
(50, -70, '2020-06-09 07:39:39', 'product1', '0'),
(51, -74, '2020-06-09 07:40:25', 'product1', '0'),
(52, -73, '2020-06-09 07:40:46', 'product1', '0'),
(53, -72, '2020-06-09 07:41:07', 'product1', '0'),
(54, -72, '2020-06-09 07:41:28', 'product1', '0'),
(55, -72, '2020-06-09 07:41:49', 'product1', '0'),
(56, -73, '2020-06-09 07:42:10', 'product1', '0'),
(57, -73, '2020-06-09 07:42:56', 'product1', '0'),
(58, -72, '2020-06-09 07:43:17', 'product1', '0'),
(59, -72, '2020-06-09 07:43:38', 'product1', '0'),
(60, -73, '2020-06-09 07:43:59', 'product1', '0'),
(61, -72, '2020-06-09 07:44:20', 'product1', '0'),
(62, -73, '2020-06-09 07:44:41', 'product1', '0'),
(63, -75, '2020-06-09 07:45:52', 'product1', '0'),
(64, -74, '2020-06-09 07:46:13', 'product1', '0'),
(65, -77, '2020-06-09 07:47:52', 'product1', '0'),
(66, -75, '2020-06-09 07:48:21', 'product1', '0'),
(67, -79, '2020-06-09 07:48:42', 'product1', '0'),
(68, -83, '2020-06-09 07:49:03', 'product1', '0'),
(69, -71, '2020-06-09 07:49:49', 'product1', '0'),
(70, -71, '2020-06-09 07:50:10', 'product1', '0'),
(71, -72, '2020-06-09 07:50:31', 'product1', '0'),
(72, -72, '2020-06-09 07:50:52', 'product1', '0'),
(73, -73, '2020-06-09 07:51:13', 'product1', '0'),
(74, -71, '2020-06-09 07:52:24', 'product1', '0'),
(75, -72, '2020-06-09 07:52:45', 'product1', '0'),
(76, -71, '2020-06-09 07:53:06', 'product1', '0'),
(77, -72, '2020-06-09 07:53:27', 'product1', '0'),
(78, -72, '2020-06-09 07:53:48', 'product1', '0'),
(79, -72, '2020-06-09 07:54:34', 'product1', '0'),
(80, -70, '2020-06-09 07:54:55', 'product1', '0'),
(81, -71, '2020-06-09 07:55:16', 'product1', '0'),
(82, -70, '2020-06-09 07:56:02', 'product1', '0'),
(83, -70, '2020-06-09 07:56:23', 'product1', '0'),
(84, -71, '2020-06-09 07:56:44', 'product1', '0'),
(85, -70, '2020-06-09 07:57:05', 'product1', '0'),
(86, -72, '2020-06-09 07:57:26', 'product1', '0'),
(87, -73, '2020-06-09 07:57:47', 'product1', '0'),
(88, -78, '2020-06-09 08:00:14', 'product1', '0'),
(89, -76, '2020-06-09 08:00:35', 'product1', '0'),
(90, -70, '2020-06-09 08:00:56', 'product1', '0'),
(91, -78, '2020-06-09 08:01:17', 'product1', '0'),
(92, -68, '2020-06-09 08:01:38', 'product1', '0'),
(93, -69, '2020-06-09 08:01:58', 'product1', '0'),
(94, -69, '2020-06-09 08:02:19', 'product1', '0'),
(95, -69, '2020-06-09 08:02:40', 'product1', '0'),
(96, -69, '2020-06-09 08:03:01', 'product1', '0'),
(97, -72, '2020-06-09 08:03:22', 'product1', '0'),
(98, -70, '2020-06-09 08:03:43', 'product1', '0'),
(99, -69, '2020-06-09 08:04:04', 'product1', '0'),
(100, -71, '2020-06-09 08:06:34', 'product1', '0'),
(201, -80, '2020-06-08 15:46:54', 'product2', '0'),
(202, -81, '2020-06-08 15:47:15', 'product2', '0'),
(203, -81, '2020-06-08 15:47:36', 'product2', '0'),
(204, -82, '2020-06-08 15:47:57', 'product2', '0'),
(205, -80, '2020-06-08 15:48:17', 'product2', '0'),
(206, -80, '2020-06-08 15:48:39', 'product2', '0'),
(207, -80, '2020-06-08 15:48:59', 'product2', '0'),
(208, -79, '2020-06-08 15:49:20', 'product2', '0'),
(209, -82, '2020-06-08 15:50:32', 'product2', '0'),
(210, -86, '2020-06-08 15:51:43', 'product2', '0'),
(211, -80, '2020-06-08 15:52:29', 'product2', '0'),
(212, -78, '2020-06-08 15:52:50', 'product2', '0'),
(213, -79, '2020-06-08 15:53:11', 'product2', '0'),
(214, -79, '2020-06-08 15:53:31', 'product2', '0'),
(215, -79, '2020-06-08 15:53:52', 'product2', '0'),
(216, -80, '2020-06-08 15:54:13', 'product2', '0'),
(217, -79, '2020-06-08 15:54:37', 'product2', '0'),
(218, -78, '2020-06-08 15:54:58', 'product2', '0'),
(219, -78, '2020-06-08 15:55:19', 'product2', '0'),
(220, -81, '2020-06-08 15:56:05', 'product2', '0'),
(221, -80, '2020-06-08 15:57:16', 'product2', '0'),
(222, -79, '2020-06-08 15:57:37', 'product2', '0'),
(223, -79, '2020-06-08 15:57:58', 'product2', '0'),
(224, -81, '2020-06-08 15:58:19', 'product2', '0'),
(225, -81, '2020-06-08 15:59:30', 'product2', '0'),
(226, -86, '2020-06-08 16:00:19', 'product2', '0'),
(227, -87, '2020-06-08 16:00:41', 'product2', '0'),
(228, -87, '2020-06-08 16:01:31', 'product2', '0'),
(229, -86, '2020-06-08 16:01:52', 'product2', '0'),
(230, -87, '2020-06-08 16:02:16', 'product2', '0'),
(231, -79, '2020-06-08 16:03:03', 'product2', '0'),
(232, -77, '2020-06-08 16:03:24', 'product2', '0'),
(233, -76, '2020-06-08 16:03:44', 'product2', '0'),
(234, -80, '2020-06-08 16:04:05', 'product2', '0'),
(235, -78, '2020-06-08 16:04:27', 'product2', '0'),
(236, -80, '2020-06-08 16:06:04', 'product2', '0'),
(237, -74, '2020-06-08 16:06:50', 'product2', '0'),
(238, -76, '2020-06-08 16:07:11', 'product2', '0'),
(239, -76, '2020-06-08 16:07:31', 'product2', '0'),
(240, -75, '2020-06-08 16:09:08', 'product2', '0'),
(241, -76, '2020-06-08 16:10:20', 'product2', '0'),
(242, -74, '2020-06-08 16:10:41', 'product2', '0'),
(243, -76, '2020-06-08 16:11:01', 'product2', '0'),
(244, -75, '2020-06-08 16:12:13', 'product2', '0'),
(245, -75, '2020-06-08 16:12:34', 'product2', '0'),
(246, -75, '2020-06-08 16:12:55', 'product2', '0'),
(247, -75, '2020-06-08 16:13:15', 'product2', '0'),
(248, -74, '2020-06-08 16:13:36', 'product2', '0'),
(249, -75, '2020-06-09 07:39:19', 'product2', '0'),
(250, -70, '2020-06-09 07:39:39', 'product2', '0'),
(251, -74, '2020-06-09 07:40:25', 'product2', '0'),
(252, -73, '2020-06-09 07:40:46', 'product2', '0'),
(253, -72, '2020-06-09 07:41:07', 'product2', '0'),
(254, -72, '2020-06-09 07:41:28', 'product2', '0'),
(255, -72, '2020-06-09 07:41:49', 'product2', '0'),
(256, -73, '2020-06-09 07:42:10', 'product2', '0'),
(257, -73, '2020-06-09 07:42:56', 'product2', '0'),
(258, -72, '2020-06-09 07:43:17', 'product2', '0'),
(259, -72, '2020-06-09 07:43:38', 'product2', '0'),
(260, -73, '2020-06-09 07:43:59', 'product2', '0'),
(261, -72, '2020-06-09 07:44:20', 'product2', '0'),
(262, -73, '2020-06-09 07:44:41', 'product2', '0'),
(263, -75, '2020-06-09 07:45:52', 'product2', '0'),
(264, -74, '2020-06-09 07:46:13', 'product2', '0'),
(265, -77, '2020-06-09 07:47:52', 'product2', '0'),
(266, -75, '2020-06-09 07:48:21', 'product2', '0'),
(267, -79, '2020-06-09 07:48:42', 'product2', '0'),
(268, -83, '2020-06-09 07:49:03', 'product2', '0'),
(269, -71, '2020-06-09 07:49:49', 'product2', '0'),
(270, -71, '2020-06-09 07:50:10', 'product2', '0'),
(271, -72, '2020-06-09 07:50:31', 'product2', '0'),
(272, -72, '2020-06-09 07:50:52', 'product2', '0'),
(273, -73, '2020-06-09 07:51:13', 'product2', '0'),
(274, -71, '2020-06-09 07:52:24', 'product2', '0'),
(275, -72, '2020-06-09 07:52:45', 'product2', '0'),
(276, -71, '2020-06-09 07:53:06', 'product2', '0'),
(277, -72, '2020-06-09 07:53:27', 'product2', '0'),
(278, -72, '2020-06-09 07:53:48', 'product2', '0'),
(279, -72, '2020-06-09 07:54:34', 'product2', '0'),
(280, -70, '2020-06-09 07:54:55', 'product2', '0'),
(281, -71, '2020-06-09 07:55:16', 'product2', '0'),
(282, -70, '2020-06-09 07:56:02', 'product2', '0'),
(283, -70, '2020-06-09 07:56:23', 'product2', '0'),
(284, -71, '2020-06-09 07:56:44', 'product2', '0'),
(285, -70, '2020-06-09 07:57:05', 'product2', '0'),
(286, -72, '2020-06-09 07:57:26', 'product2', '0'),
(287, -73, '2020-06-09 07:57:47', 'product2', '0'),
(288, -78, '2020-06-09 08:00:14', 'product2', '0'),
(289, -76, '2020-06-09 08:00:35', 'product2', '0'),
(290, -70, '2020-06-09 08:00:56', 'product2', '0'),
(291, -78, '2020-06-09 08:01:17', 'product2', '0'),
(292, -68, '2020-06-09 08:01:38', 'product2', '0'),
(293, -69, '2020-06-09 08:01:58', 'product2', '0'),
(294, -69, '2020-06-09 08:02:19', 'product2', '0'),
(295, -69, '2020-06-09 08:02:40', 'product2', '0'),
(296, -69, '2020-06-09 08:03:01', 'product2', '0'),
(297, -72, '2020-06-09 08:03:22', 'product2', '0'),
(298, -70, '2020-06-09 08:03:43', 'product2', '0'),
(299, -69, '2020-06-09 08:04:04', 'product2', '0'),
(300, -71, '2020-06-09 08:06:34', 'product2', '0');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `accounts`
--
ALTER TABLE `accounts`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `diagnoses_ml`
--
ALTER TABLE `diagnoses_ml`
  ADD PRIMARY KEY (`sno`);

--
-- Indexes for table `glucose_value`
--
ALTER TABLE `glucose_value`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `accounts`
--
ALTER TABLE `accounts`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `diagnoses_ml`
--
ALTER TABLE `diagnoses_ml`
  MODIFY `sno` int(50) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=24;

--
-- AUTO_INCREMENT for table `glucose_value`
--
ALTER TABLE `glucose_value`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=301;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
