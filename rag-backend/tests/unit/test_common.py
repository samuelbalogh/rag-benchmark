import unittest
from unittest.mock import patch, MagicMock
import datetime
from jose import jwt

from common.auth import create_api_key, validate_api_key
from common.errors import AuthenticationError, RAGError
from common.vector import cosine_similarity
import numpy as np


class TestCommonUtils(unittest.TestCase):
    def test_cosine_similarity_happy_path(self):
        # arrange
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 1, 0])
        
        # act
        sim1_2 = cosine_similarity(vec1, vec2)
        sim1_3 = cosine_similarity(vec1, vec3)
        sim2_3 = cosine_similarity(vec2, vec3)
        sim1_1 = cosine_similarity(vec1, vec1)
        
        # assert
        self.assertEqual(sim1_2, 0.0)  # orthogonal vectors
        self.assertAlmostEqual(sim1_3, 0.7071067811865475)  # 45 degree angle
        self.assertAlmostEqual(sim2_3, 0.7071067811865475)  # 45 degree angle
        self.assertEqual(sim1_1, 1.0)  # same vector
    
    @patch('common.auth.jwt.encode')
    @patch('common.auth.datetime')
    @patch('common.auth.uuid.uuid4')
    def test_create_api_key_happy_path(self, mock_uuid, mock_datetime, mock_jwt_encode):
        # arrange
        mock_uuid.return_value = MagicMock(hex="test-uuid")
        mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta
        mock_jwt_encode.return_value = "test-jwt-token"
        
        # act
        result = create_api_key("test-user", "test-key-name")
        
        # assert
        mock_jwt_encode.assert_called_once()
        self.assertEqual(result["key"], "test-jwt-token")
        self.assertEqual(result["user_id"], "test-user")
        self.assertEqual(result["name"], "test-key-name")
        self.assertEqual(result["created_at"], mock_now)
        self.assertEqual(result["key_id"], "test-uuid")
    
    @patch('common.auth.jwt.decode')
    def test_validate_api_key_happy_path(self, mock_jwt_decode):
        # arrange
        mock_jwt_decode.return_value = {
            "sub": "test-user",
            "key_id": "test-key-id",
            "exp": datetime.datetime.now().timestamp() + 3600
        }
        
        # act
        result = validate_api_key("test-jwt-token", "test-secret")
        
        # assert
        mock_jwt_decode.assert_called_once_with(
            "test-jwt-token", "test-secret", algorithms=["HS256"])
        self.assertEqual(result["user_id"], "test-user")
        self.assertEqual(result["key_id"], "test-key-id")
    
    @patch('common.auth.jwt.decode')
    def test_validate_api_key_expired(self, mock_jwt_decode):
        # arrange
        mock_jwt_decode.side_effect = jwt.ExpiredSignatureError
        
        # act and assert
        with self.assertRaises(AuthenticationError):
            validate_api_key("test-jwt-token", "test-secret")
    
    def test_rag_error_happy_path(self):
        # arrange
        error_message = "Test error message"
        error_code = "TEST_ERROR"
        
        # act
        error = RAGError(message=error_message, code=error_code)
        
        # assert
        self.assertEqual(str(error), error_message)
        self.assertEqual(error.code, error_code)
        self.assertEqual(error.status_code, 500)  # default status code 