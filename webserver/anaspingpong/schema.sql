--DROP TABLE IF EXISTS tables;

CREATE TABLE IF NOT EXISTS tables (
  hash INT PRIMARY KEY,
  latitude FLOAT NOT NULL,
  longitude FLOAT NOT NULL
);
