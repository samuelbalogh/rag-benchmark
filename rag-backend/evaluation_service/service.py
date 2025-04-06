"""Evaluation service for measuring RAG performance metrics."""

import time
import numpy as np
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from vector_store.models import Document
from common.logging import get_logger
from common.config import get_settings
from evaluation_service.metrics import calculate_relevance, calculate_truthfulness, calculate_completeness
from evaluation_service.utils import get_latency, count_tokens, get_benchmark_questions, process_query, save_benchmark_results

# Initialize logger
logger = get_logger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=get_settings().openai_api_key)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    score: float
    max_score: float = 1.0
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def calculate_metrics(
    query: str,
    response: str,
    retrieved_documents: List[Document],
    ground_truth: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate evaluation metrics for a RAG response.
    
    Args:
        query: Original query
        response: Generated response
        retrieved_documents: Documents retrieved by the system
        ground_truth: Optional ground truth answer
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with metrics results
    """
    if metrics is None:
        metrics = ["relevance", "faithfulness", "context_recall", "latency"]
    
    results = {}
    
    # Process each requested metric
    for metric in metrics:
        if metric == "relevance":
            results[metric] = evaluate_relevance(query, response).score
        elif metric == "faithfulness":
            results[metric] = evaluate_faithfulness(response, retrieved_documents).score
        elif metric == "context_recall":
            if ground_truth:
                results[metric] = evaluate_context_recall(retrieved_documents, ground_truth).score
            else:
                results[metric] = None
        elif metric == "answer_relevancy":
            results[metric] = evaluate_answer_relevancy(query, response).score
        elif metric == "latency":
            # This is usually captured during response generation
            # Here we just include a placeholder
            results[metric] = 0.0
    
    return results


def evaluate_relevance(query: str, response: str) -> EvaluationResult:
    """Evaluate the relevance of a response to the query.
    
    Args:
        query: Original query
        response: Generated response
        
    Returns:
        EvaluationResult with score, explanation and metadata
    """
    try:
        # Simple TF-IDF based relevance score
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query, response])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Normalize to 0-1 range
        score = float(min(max(similarity, 0.0), 1.0))
        
        return EvaluationResult(
            score=score,
            explanation=f"Relevance score: {score:.2f} based on TF-IDF cosine similarity",
            metadata={"method": "tfidf_cosine_similarity"}
        )
    except Exception as e:
        logger.error(f"Error evaluating relevance: {str(e)}")
        return EvaluationResult(
            score=0.0,
            explanation=f"Error during relevance evaluation: {str(e)}",
            metadata={"error": str(e)}
        )


def evaluate_faithfulness(response: str, documents: List[Document]) -> EvaluationResult:
    """Evaluate the faithfulness of a response to the retrieved documents.
    
    Args:
        response: Generated response
        documents: Retrieved documents
        
    Returns:
        EvaluationResult with score, explanation and metadata
    """
    try:
        # Extract key statements from response
        statements = _extract_statements(response)
        
        # If no statements were found, return a minimum score
        if not statements:
            return EvaluationResult(
                score=0.0,
                explanation="No statements found in response",
                metadata={"statement_count": 0}
            )
        
        # Combine document content
        context = "\n\n".join([doc.content for doc in documents])
        
        # Check each statement against the context
        supported_count = 0
        for statement in statements:
            if _is_statement_supported(statement, context):
                supported_count += 1
        
        # Calculate faithfulness score
        score = supported_count / len(statements) if statements else 0.0
        
        return EvaluationResult(
            score=score,
            explanation=f"Faithfulness score: {score:.2f} ({supported_count}/{len(statements)} statements supported)",
            metadata={"statement_count": len(statements), "supported_count": supported_count}
        )
    except Exception as e:
        logger.error(f"Error evaluating faithfulness: {str(e)}")
        return EvaluationResult(
            score=0.0,
            explanation=f"Error during faithfulness evaluation: {str(e)}",
            metadata={"error": str(e)}
        )


def _extract_statements(text: str) -> List[str]:
    """Extract factual statements from text.
    
    Args:
        text: Text to extract statements from
        
    Returns:
        List of extracted statements
    """
    # Simple heuristic: split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out short sentences and questions
    statements = [
        sentence.strip() 
        for sentence in sentences 
        if len(sentence.strip()) > 20 and not sentence.strip().endswith("?")
    ]
    
    return statements


def _is_statement_supported(statement: str, context: str) -> bool:
    """Check if a statement is supported by the context.
        
        Args:
        statement: Statement to check
        context: Context text
            
        Returns:
        True if statement is supported, False otherwise
    """
    # Simple heuristic: check for keyword overlap
    statement_words = set(statement.lower().split())
    context_words = set(context.lower().split())
    
    # Remove common words
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "is", "are", "was", "were"}
    statement_words = statement_words - common_words
    
    # Calculate overlap
    overlap = len(statement_words & context_words) / len(statement_words) if statement_words else 0
    
    return overlap > 0.6  # Threshold for considering a statement supported


def evaluate_context_recall(documents: List[Document], ground_truth: str) -> EvaluationResult:
    """Evaluate how well the retrieved documents cover the ground truth.
    
    Args:
        documents: Retrieved documents
        ground_truth: Ground truth answer
        
    Returns:
        EvaluationResult with score, explanation and metadata
    """
    try:
        # Combine document content
        context = "\n\n".join([doc.content for doc in documents])
        
        # Extract key facts from ground truth
        ground_truth_facts = _extract_statements(ground_truth)
        
        if not ground_truth_facts:
            return EvaluationResult(
                score=0.0,
                explanation="No facts found in ground truth",
                metadata={"fact_count": 0}
            )
        
        # Check how many facts from ground truth are covered by the context
        covered_count = 0
        for fact in ground_truth_facts:
            if _is_statement_supported(fact, context):
                covered_count += 1
        
        # Calculate recall score
        score = covered_count / len(ground_truth_facts) if ground_truth_facts else 0.0
        
        return EvaluationResult(
            score=score,
            explanation=f"Context recall score: {score:.2f} ({covered_count}/{len(ground_truth_facts)} facts covered)",
            metadata={"fact_count": len(ground_truth_facts), "covered_count": covered_count}
        )
    except Exception as e:
        logger.error(f"Error evaluating context recall: {str(e)}")
        return EvaluationResult(
            score=0.0,
            explanation=f"Error during context recall evaluation: {str(e)}",
            metadata={"error": str(e)}
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def evaluate_answer_relevancy(query: str, response: str) -> EvaluationResult:
    """Evaluate the relevancy of the answer to the query using an LLM judge.
        
        Args:
        query: Original query
        response: Generated response
            
        Returns:
        EvaluationResult with score, explanation and metadata
    """
    try:
        prompt = f"""
        Evaluate the relevancy of the following response to the query.
        Score from 0 to 10, where 0 is completely irrelevant and 10 is perfectly relevant.
        
        Query: {query}
        
        Response: {response}
        
        First, explain your reasoning for the score. Then on the last line, output only the score as a number between 0 and 10.
        """
        
        llm_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of question answering systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        evaluation_text = llm_response.choices[0].message.content
        
        # Extract score from the last line
        lines = evaluation_text.strip().split('\n')
        score_text = lines[-1].strip()
        
        # Try to extract the score as a number
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            score = float(score_match.group(1)) / 10.0  # Normalize to 0-1
        else:
            score = 0.5  # Default if we can't extract a score
            
        # Extract explanation (everything except the last line)
        explanation = '\n'.join(lines[:-1]).strip()
        
        return EvaluationResult(
            score=score,
            explanation=explanation,
            metadata={"method": "llm_judge", "raw_score_text": score_text}
        )
    except Exception as e:
        logger.error(f"Error evaluating answer relevancy: {str(e)}")
        return EvaluationResult(
            score=0.5,  # Default to middle score on error
            explanation=f"Error during answer relevancy evaluation: {str(e)}",
            metadata={"error": str(e)}
        )


def evaluate_response(
    query: str,
    response: str,
    retrieved_documents: List[Document],
    ground_truth: Optional[str] = None,
    methods: Optional[List[str]] = None
) -> Dict[str, EvaluationResult]:
    """Evaluate a response using multiple evaluation methods.
    
    Args:
        query: Original query
        response: Generated response
        retrieved_documents: Documents retrieved by the system
        ground_truth: Optional ground truth answer
        methods: List of evaluation methods to use
            
        Returns:
        Dictionary mapping method names to EvaluationResult objects
    """
    if methods is None:
        methods = ["relevance", "faithfulness"]
        if ground_truth:
            methods.append("context_recall")
    
    results = {}
    
    for method in methods:
        try:
            if method == "relevance":
                results[method] = evaluate_relevance(query, response)
            elif method == "faithfulness":
                results[method] = evaluate_faithfulness(response, retrieved_documents)
            elif method == "context_recall" and ground_truth:
                results[method] = evaluate_context_recall(retrieved_documents, ground_truth)
            elif method == "answer_relevancy":
                results[method] = evaluate_answer_relevancy(query, response)
            else:
                logger.warning(f"Unknown evaluation method: {method}")
        except Exception as e:
            logger.error(f"Error evaluating with method {method}: {str(e)}")
            results[method] = EvaluationResult(
                score=0.0,
                explanation=f"Error during {method} evaluation: {str(e)}",
                metadata={"error": str(e)}
            )
    
    return results


class EvaluationService:
    """Service for evaluating RAG system performance."""
    
    def __init__(self, db=None):
        """Initialize the evaluation service.
        
        Args:
            db: Optional database session
        """
        self.db = db
        self.logger = get_logger(__name__)
    
    @staticmethod
    def calculate_metrics(query: str, context: List[str], response: str) -> Dict[str, Any]:
        """Calculate evaluation metrics for a RAG response.
        
        Args:
            query: Original query
            context: Retrieved context passages
            response: Generated response
            
        Returns:
            Dictionary with metrics results
        """
        try:
            # Calculate individual metrics
            relevance = calculate_relevance(query, context, response)
            truthfulness = calculate_truthfulness(context, response)
            completeness = calculate_completeness(query, response)
            latency = get_latency()
            tokens = count_tokens(context, response)
            
            return {
                "relevance": relevance,
                "truthfulness": truthfulness,
                "completeness": completeness,
                "latency_ms": latency,
                "tokens": tokens
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "error": str(e)
            }
    
    @staticmethod
    def run_benchmark(document_ids: List[str], strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark evaluation on a set of test questions.
        
        Args:
            document_ids: List of document IDs to use for retrieval
            strategy_name: Name of the retrieval strategy to test
            parameters: Strategy parameters
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            # Get benchmark questions
            questions = get_benchmark_questions()
            results = []
            
            # Process each question
            for question in questions:
                result = process_query(question["id"], question["question"], 
                                    document_ids, strategy_name, parameters)
                results.append(result)
            
            # Calculate aggregate metrics
            avg_relevance = np.mean([r["metrics"]["relevance"] for r in results])
            avg_completeness = np.mean([r["metrics"]["completeness"] for r in results])
            
            benchmark_results = {
                "total_questions": len(questions),
                "avg_relevance": float(avg_relevance),
                "avg_completeness": float(avg_completeness),
                "detailed_results": results
            }
            
            # Save results
            save_benchmark_results(benchmark_results)
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            return {
                "error": str(e)
            } 