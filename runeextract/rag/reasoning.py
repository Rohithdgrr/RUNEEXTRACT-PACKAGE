"""
Feature 14: Chain-of-Thought Reasoning

Multi-step reasoning for complex RAG queries.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Single step in reasoning chain."""
    step_number: int
    question: str
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str = ""


@dataclass
class ReasoningTrace:
    """Complete reasoning trace."""
    original_query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_latency_ms: float = 0.0
    
    def add_step(
        self,
        step_number: int,
        question: str,
        answer: str,
        confidence: float,
        sources: Optional[List[str]] = None,
        reasoning: str = ""
    ) -> None:
        """Add a reasoning step."""
        self.steps.append(ReasoningStep(
            step_number=step_number,
            question=question,
            answer=answer,
            confidence=confidence,
            sources=sources or [],
            reasoning=reasoning
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "steps": [
                {
                    "step": s.step_number,
                    "question": s.question,
                    "answer": s.answer,
                    "confidence": s.confidence,
                    "sources": s.sources,
                    "reasoning": s.reasoning
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "total_latency_ms": self.total_latency_ms
        }


class ChainOfThoughtReasoner:
    """Chain-of-thought reasoning for complex RAG queries.
    
    Features:
    - Automatic query decomposition
    - Multi-step reasoning with context accumulation
    - Self-correction (re-query if confidence low)
    - Reasoning trace for transparency
    - Multi-hop question answering
    
    Usage::
    
        reasoner = ChainOfThoughtReasoner(rag)
        
        # Complex multi-part query
        result = reasoner.reason(
            "Compare the revenue growth rates in Q1 vs Q2, "
            "and explain the factors that caused the difference."
        )
        
        # Show reasoning trace
        for step in result.trace.steps:
            print(f"Step {step.step_number}: {step.question}")
            print(f"Answer: {step.answer}\n")
        
        print(f"Final: {result.trace.final_answer}")
    """
    
    def __init__(
        self,
        rag: Any,
        max_steps: int = 5,
        confidence_threshold: float = 0.7,
        enable_self_correction: bool = True
    ):
        """Initialize chain-of-thought reasoner.
        
        Args:
            rag: Base AutoRAG instance
            max_steps: Maximum reasoning steps
            confidence_threshold: Min confidence before self-correction
            enable_self_correction: Enable re-querying on low confidence
        """
        self.rag = rag
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.enable_self_correction = enable_self_correction
        
        logger.info(
            f"ChainOfThoughtReasoner initialized "
            f"(max_steps={max_steps}, threshold={confidence_threshold:.2%})"
        )
    
    def decompose_query(self, query: str) -> List[str]:
        """Break complex query into sub-questions.
        
        Args:
            query: Complex query
        
        Returns:
            List of sub-questions
        """
        prompt = f"""Break this complex question into 3-5 simpler sub-questions that, 
when answered in sequence, will help answer the original question.

Original question: {query}

List the sub-questions (one per line, no numbering):"""
        
        response = self.rag.ai.call(prompt)
        
        # Parse sub-questions
        sub_questions = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        
        # Limit to max_steps
        sub_questions = sub_questions[:self.max_steps]
        
        logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
        return sub_questions
    
    def reason(self, query: str, **kwargs) -> Any:
        """Perform chain-of-thought reasoning.
        
        Args:
            query: Complex query requiring multi-step reasoning
            **kwargs: Passed to rag.query()
        
        Returns:
            RAGResult with reasoning_trace attribute
        """
        import time
        
        start = time.time()
        
        # Create trace
        trace = ReasoningTrace(original_query=query)
        
        # Decompose query
        sub_questions = self.decompose_query(query)
        
        # Accumulated context from previous steps
        context_parts = []
        
        # Process each sub-question
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info(f"Reasoning step {i}/{len(sub_questions)}: {sub_q}")
            
            # Query RAG
            result = self.rag.query(sub_q, **kwargs)
            
            # Extract sources
            sources = list(set(
                chunk.source
                for chunk in result.retrieved_chunks
                if chunk.source
            ))
            
            # Add to trace
            trace.add_step(
                step_number=i,
                question=sub_q,
                answer=result.answer,
                confidence=result.confidence,
                sources=sources[:3],  # Top 3 sources
                reasoning=f"Retrieved {len(result.retrieved_chunks)} chunks"
            )
            
            # Accumulate context
            context_parts.append(f"Q{i}: {sub_q}\nA{i}: {result.answer}")
            
            # Self-correction if confidence too low
            if (self.enable_self_correction and 
                result.confidence < self.confidence_threshold):
                logger.info(f"Low confidence ({result.confidence:.2%}), re-querying with more context")
                
                # Re-query with accumulated context
                context_prompt = "\n\n".join(context_parts[:-1])
                refined_q = f"Context:\n{context_prompt}\n\nQuestion: {sub_q}"
                
                result_refined = self.rag.query(refined_q, **kwargs)
                
                if result_refined.confidence > result.confidence:
                    logger.info(f"Improved confidence: {result.confidence:.2%} → {result_refined.confidence:.2%}")
                    trace.steps[-1].answer = result_refined.answer
                    trace.steps[-1].confidence = result_refined.confidence
                    trace.steps[-1].reasoning += " (self-corrected)"
                    context_parts[-1] = f"Q{i}: {sub_q}\nA{i}: {result_refined.answer}"
        
        # Synthesize final answer
        trace.final_answer = self._synthesize_answer(query, context_parts)
        
        # Compute final confidence (weighted average of step confidences)
        if trace.steps:
            # Weight later steps more heavily
            weights = [i + 1 for i in range(len(trace.steps))]
            total_weight = sum(weights)
            trace.final_confidence = sum(
                s.confidence * w
                for s, w in zip(trace.steps, weights)
            ) / total_weight
        
        trace.total_latency_ms = (time.time() - start) * 1000
        
        logger.info(
            f"Reasoning complete: {len(trace.steps)} steps, "
            f"confidence={trace.final_confidence:.2%}"
        )
        
        # Create result object
        from runeextract.rag.types import RAGResult
        
        result = RAGResult(
            answer=trace.final_answer,
            citations=[],
            confidence=trace.final_confidence,
            retrieved_chunks=[],
            query_variants=[],
            latency_ms=trace.total_latency_ms,
            tokens_used={"input": 0, "output": 0},
            cost=0.0,
            total_session_cost=0.0
        )
        
        # Attach reasoning trace
        result.reasoning_trace = trace
        
        return result
    
    def _synthesize_answer(
        self,
        original_query: str,
        context_parts: List[str]
    ) -> str:
        """Synthesize final answer from reasoning steps.
        
        Args:
            original_query: Original question
            context_parts: List of Q&A pairs from reasoning steps
        
        Returns:
            Synthesized answer
        """
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the step-by-step reasoning below, provide a comprehensive answer 
to the original question. Synthesize the information from all steps.

Original question: {original_query}

Step-by-step reasoning:
{context}

Comprehensive answer:"""
        
        response = self.rag.ai.call(prompt)
        return response.strip()
    
    def reason_with_reflection(self, query: str, **kwargs) -> Any:
        """Reason with self-reflection loop.
        
        Similar to reason(), but adds a reflection step where the reasoner
        reviews its own answer and refines it if needed.
        
        Args:
            query: Complex query
            **kwargs: Passed to rag.query()
        
        Returns:
            RAGResult with reasoning_trace
        """
        # Initial reasoning
        result = self.reason(query, **kwargs)
        
        # Reflection prompt
        reflection_prompt = f"""Review this answer and identify any gaps, errors, 
or areas that need clarification:

Question: {query}
Answer: {result.answer}

Issues found:"""
        
        reflection = self.rag.ai.call(reflection_prompt)
        
        # If issues found, refine
        if len(reflection.strip()) > 20:  # Not just "None" or "No issues"
            logger.info("Reflection found issues, refining answer")
            
            refine_prompt = f"""Improve this answer based on the review:

Question: {query}
Original answer: {result.answer}
Review: {reflection}

Improved answer:"""
            
            refined = self.rag.ai.call(refine_prompt)
            result.answer = refined.strip()
            
            # Add reflection to trace
            if hasattr(result, 'reasoning_trace'):
                result.reasoning_trace.steps.append(
                    ReasoningStep(
                        step_number=len(result.reasoning_trace.steps) + 1,
                        question="Reflection",
                        answer=reflection,
                        confidence=1.0,
                        sources=[],
                        reasoning="Self-reflection and refinement"
                    )
                )
        
        return result
