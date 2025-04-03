import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

# assuming this is the structure, update imports if needed
from knowledge_graph_service.service import build_knowledge_graph, query_knowledge_graph
from knowledge_graph_service.entity_extraction import extract_entities


class TestKnowledgeGraphService(unittest.TestCase):
    @patch('knowledge_graph_service.service.get_document_chunks')
    @patch('knowledge_graph_service.service.extract_entities')
    @patch('knowledge_graph_service.service.build_relationships')
    @patch('knowledge_graph_service.service.save_graph')
    @patch('knowledge_graph_service.service.update_document_status')
    def test_build_knowledge_graph_happy_path(self, mock_update_status, mock_save_graph,
                                           mock_build_relations, mock_extract_entities, 
                                           mock_get_chunks):
        # arrange
        document_id = "test-doc-id"
        chunks = [
            {"id": "chunk1", "text": "Alice visited London with Bob", "position": 0},
            {"id": "chunk2", "text": "Bob works at Google in London", "position": 1},
        ]
        mock_get_chunks.return_value = chunks
        
        entities = [
            {"id": "e1", "text": "Alice", "type": "PERSON", "chunk_id": "chunk1"},
            {"id": "e2", "text": "London", "type": "LOCATION", "chunk_id": "chunk1"},
            {"id": "e3", "text": "Bob", "type": "PERSON", "chunk_id": "chunk1"},
            {"id": "e4", "text": "Bob", "type": "PERSON", "chunk_id": "chunk2"},
            {"id": "e5", "text": "Google", "type": "ORGANIZATION", "chunk_id": "chunk2"},
            {"id": "e6", "text": "London", "type": "LOCATION", "chunk_id": "chunk2"},
        ]
        mock_extract_entities.return_value = entities
        
        # create a sample graph
        graph = nx.DiGraph()
        graph.add_nodes_from(["Alice", "Bob", "London", "Google"])
        graph.add_edges_from([
            ("Alice", "London", {"relation": "visited"}),
            ("Alice", "Bob", {"relation": "with"}),
            ("Bob", "Google", {"relation": "works_at"}),
            ("Google", "London", {"relation": "in"})
        ])
        mock_build_relations.return_value = graph
        
        # act
        result = build_knowledge_graph(document_id)
        
        # assert
        mock_get_chunks.assert_called_once_with(document_id)
        mock_extract_entities.assert_called_once()
        mock_build_relations.assert_called_once_with(entities)
        mock_save_graph.assert_called_once()
        mock_update_status.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document_id"], document_id)
        self.assertEqual(result["node_count"], 4)
        self.assertEqual(result["edge_count"], 4)
    
    @patch('knowledge_graph_service.entity_extraction.spacy_nlp')
    def test_extract_entities_happy_path(self, mock_spacy_nlp):
        # arrange
        text = "Alice visited London with Bob who works at Google."
        
        # create mock spacy entities
        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 5
        
        mock_ent2 = MagicMock()
        mock_ent2.text = "London"
        mock_ent2.label_ = "GPE"  # spacy uses GPE for geo-political entities
        mock_ent2.start_char = 14
        mock_ent2.end_char = 20
        
        mock_ent3 = MagicMock()
        mock_ent3.text = "Bob"
        mock_ent3.label_ = "PERSON"
        mock_ent3.start_char = 26
        mock_ent3.end_char = 29
        
        mock_ent4 = MagicMock()
        mock_ent4.text = "Google"
        mock_ent4.label_ = "ORG"
        mock_ent4.start_char = 41
        mock_ent4.end_char = 47
        
        # mock document processing
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3, mock_ent4]
        mock_spacy_nlp.return_value = mock_doc
        
        # act
        result = extract_entities(text, chunk_id="chunk1")
        
        # assert
        mock_spacy_nlp.assert_called_once_with(text)
        self.assertEqual(len(result), 4)
        
        # check entity details
        entity_texts = [entity["text"] for entity in result]
        self.assertIn("Alice", entity_texts)
        self.assertIn("London", entity_texts)
        self.assertIn("Bob", entity_texts)
        self.assertIn("Google", entity_texts)
        
        for entity in result:
            self.assertIn("id", entity)
            self.assertIn("type", entity)
            self.assertEqual(entity["chunk_id"], "chunk1")
    
    @patch('knowledge_graph_service.service.load_graph')
    def test_query_knowledge_graph_happy_path(self, mock_load_graph):
        # arrange
        query = "What is the relationship between Alice and London?"
        document_id = "test-doc-id"
        
        # create a sample graph
        graph = nx.DiGraph()
        graph.add_nodes_from(["Alice", "Bob", "London", "Google"])
        graph.add_edges_from([
            ("Alice", "London", {"relation": "visited"}),
            ("Alice", "Bob", {"relation": "with"}),
            ("Bob", "Google", {"relation": "works_at"}),
            ("Google", "London", {"relation": "in"})
        ])
        mock_load_graph.return_value = graph
        
        # act
        result = query_knowledge_graph(query, document_id)
        
        # assert
        mock_load_graph.assert_called_once_with(document_id)
        self.assertEqual(result["status"], "success")
        self.assertIn("subgraph", result)
        self.assertIn("relationships", result)
        
        # check if the right relationship was found
        self.assertIn({"source": "Alice", "target": "London", "relation": "visited"}, 
                    result["relationships"]) 