-- Spark 학습을 위한 데이터베이스 초기화 스크립트

-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS sparkdb;
USE sparkdb;

-- 샘플 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 샘플 데이터 삽입
INSERT INTO users (name, age, city) VALUES
('Alice', 25, 'Seoul'),
('Bob', 30, 'Busan'),
('Charlie', 35, 'Incheon'),
('Diana', 28, 'Daegu'),
('Eve', 32, 'Daejeon');

-- 제품 테이블 생성
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2),
    category VARCHAR(100),
    stock INT DEFAULT 0
);

-- 제품 샘플 데이터
INSERT INTO products (name, price, category, stock) VALUES
('Laptop', 1200000.00, 'Electronics', 50),
('Smartphone', 800000.00, 'Electronics', 100),
('Book', 25000.00, 'Books', 200),
('Coffee', 5000.00, 'Food', 500),
('T-shirt', 30000.00, 'Clothing', 150);

-- 주문 테이블 생성
CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    product_id INT,
    quantity INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- 주문 샘플 데이터
INSERT INTO orders (user_id, product_id, quantity) VALUES
(1, 1, 1),
(2, 2, 2),
(3, 3, 5),
(1, 4, 10),
(4, 5, 3);

-- 권한 설정
GRANT ALL PRIVILEGES ON sparkdb.* TO 'sparkuser'@'%';
FLUSH PRIVILEGES;

-- 테이블 확인
SHOW TABLES;
SELECT 'Database initialization completed!' as status; 