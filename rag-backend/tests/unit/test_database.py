import unittest
from unittest.mock import patch, MagicMock

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String

from common.database import get_db_session, init_db, Base


# Create a test model
class TestModel(Base):
    __tablename__ = "test_model"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)


class TestDatabase(unittest.TestCase):
    @patch('common.database.create_engine')
    @patch('common.database.sessionmaker')
    def test_get_db_session_happy_path(self, mock_sessionmaker, mock_create_engine):
        # arrange
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mock_session_class = MagicMock()
        mock_session = MagicMock(spec=Session)
        mock_session_class.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_class
        
        # act
        db_url = "postgresql://user:password@localhost:5432/db"
        result = get_db_session(db_url)
        
        # assert
        mock_create_engine.assert_called_once_with(db_url)
        mock_sessionmaker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine)
        self.assertEqual(result, mock_session)
    
    @patch('common.database.create_engine')
    def test_init_db_happy_path(self, mock_create_engine):
        # arrange
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # act
        db_url = "postgresql://user:password@localhost:5432/db"
        init_db(db_url)
        
        # assert
        mock_create_engine.assert_called_once_with(db_url)
        mock_engine.connect.assert_called_once()
        self.assertEqual(Base.metadata.create_all.call_count, 1)
        
        # verify that the model is registered
        self.assertIn(TestModel.__tablename__, Base.metadata.tables) 