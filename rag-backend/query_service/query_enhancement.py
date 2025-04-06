"""Query enhancement techniques for the RAG benchmark platform."""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import re

import nltk
from nltk.corpus import wordnet
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from common.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')


class QueryEnhancer:
    """Class for enhancing queries using various techniques."""
    
    def __init__(self):
        """Initialize the query enhancer."""
        pass
    
    def synonym_expansion(self, query: str) -> List[str]:
        """Expand query with synonyms of key terms.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded queries
        """
        # Tokenize query
        tokens = nltk.word_tokenize(query)
        expanded_queries = [query]
        
        # Process each token
        for i, token in enumerate(tokens):
            # Skip short words and stopwords
            if len(token) <= 3 or token.lower() in {'the', 'and', 'or', 'of', 'in', 'to', 'a', 'is', 'that', 'it', 'for'}:
                continue
            
            # Get synonyms
            synonyms = self._get_synonyms(token)
            
            # Create expanded queries
            for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                if synonym != token:
                    new_tokens = tokens.copy()
                    new_tokens[i] = synonym
                    expanded_queries.append(' '.join(new_tokens))
        
        return expanded_queries[:5]  # Limit to 5 expanded queries
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet.
        
        Args:
            word: Word to find synonyms for
            
        Returns:
            List of synonyms
        """
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical perfect document for the query using LLM.
        
        Args:
            query: The original query
            
        Returns:
            Hypothetical document text
        """
        try:
            prompt = f"""
            Create a detailed, informative document that would perfectly answer the following question:
            
            Question: {query}
            
            Provide a comprehensive document that includes all information needed to answer this question thoroughly.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research assistant. Create a detailed and informative document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            # Fallback to a simpler approach if the API call fails
            return f"This document addresses {query} with comprehensive information about all related concepts and details."
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def query_decomposition(self, query: str) -> List[str]:
        """Break complex queries into simpler subqueries using LLM.
        
        Args:
            query: The original query
            
        Returns:
            List of subqueries
        """
        try:
            prompt = f"""
            Break down the following complex question into 2-4 simpler, more specific questions:
            
            Complex question: {query}
            
            Output only the list of simpler questions, one per line, with no additional explanation.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at breaking down complex questions into simpler components."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse the response to get individual questions
            text = response.choices[0].message.content
            subqueries = [q.strip() for q in text.split('\n') if q.strip()]
            
            # Ensure we have at least one subquery
            if not subqueries:
                return [query]
            
            return subqueries
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}")
            # Fallback to a simple decomposition
            return [query]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def llm_rewrite(self, query: str) -> str:
        """Use LLM to rephrase query for better retrieval.
        
        Args:
            query: The original query
            
        Returns:
            Rephrased query
        """
        try:
            prompt = f"""
            Rephrase the following question to make it more specific, detailed, and optimized for a retrieval system:
            
            Original question: {query}
            
            Output only the rephrased question with no additional explanation.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at rephrasing questions to make them more specific and information-rich."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error rewriting query with LLM: {str(e)}")
            # Fallback to original query
            return query
    
    def rewrite_with_llm(self, query: str) -> str:
        """Alias for llm_rewrite for backward compatibility.
        
        Args:
            query: The original query
            
        Returns:
            Rephrased query
        """
        return self.llm_rewrite(query)


def rewrite_with_llm(query: str) -> str:
    """Use LLM to rephrase query for better retrieval.
    
    Args:
        query: The original query
        
    Returns:
        Rephrased query
    """
    enhancer = QueryEnhancer()
    return enhancer.llm_rewrite(query)


def enhance_query(query: str, enhancement_method: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
    """Enhance a query using specified enhancement method.
    
    Args:
        query: The original query
        enhancement_method: Name of enhancement method to use
        parameters: Optional parameters for enhancement
        
    Returns:
        Enhanced query or list of queries, depending on the method
    """
    if parameters is None:
        parameters = {}
    
    start_time = time.time()
    enhancer = QueryEnhancer()
    
    # Select enhancement method based on specified name
    try:
        if enhancement_method == "synonym_expansion":
            result = enhancer.synonym_expansion(query)
            logger.info(f"Synonym expansion generated {len(result)} variants in {time.time() - start_time:.2f}s")
            return result
        elif enhancement_method == "hypothetical_document":
            result = enhancer.hypothetical_document(query)
            logger.info(f"Hypothetical document generated in {time.time() - start_time:.2f}s")
            return result
        elif enhancement_method == "query_decomposition":
            result = enhancer.query_decomposition(query)
            logger.info(f"Query decomposed into {len(result)} subqueries in {time.time() - start_time:.2f}s")
            return result
        elif enhancement_method == "llm_rewrite":
            result = enhancer.llm_rewrite(query)
            logger.info(f"Query rewritten with LLM in {time.time() - start_time:.2f}s")
            return result
        else:
            # Default to returning original query if no valid method is specified
            logger.info(f"No enhancement applied, using original query")
            return query
    except Exception as e:
        logger.error(f"Error in query enhancement: {str(e)}")
        return query


def combine_enhancement_methods(query: str, methods: List[str]) -> Dict[str, Any]:
    """Apply multiple enhancement methods to a query.
    
    Args:
        query: The original query
        methods: List of enhancement methods to apply
        
    Returns:
        Dictionary with results from each method
    """
    results = {}
    enhancer = QueryEnhancer()
    
    for method in methods:
        if method == "synonym_expansion":
            results[method] = enhancer.synonym_expansion(query)
        elif method == "hypothetical_document":
            results[method] = enhancer.hypothetical_document(query)
        elif method == "query_decomposition":
            results[method] = enhancer.query_decomposition(query)
        elif method == "llm_rewrite":
            results[method] = enhancer.llm_rewrite(query)
    
    # Always include the original query
    results["original"] = query
    
    return results
