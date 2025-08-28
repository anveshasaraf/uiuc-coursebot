import tiktoken
import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from config import Config

class TokenAwarePromptOptimizer:
    """Token-aware prompt optimization for efficient API usage"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_context_tokens = self._get_max_context_tokens()
        self.target_response_tokens = Config.OPENAI_MAX_TOKENS
        self.reserved_tokens = 200  # Reserve for prompt structure and safety
        
        # Optimized prompt templates by length
        self.prompt_templates = self._create_optimized_templates()
        
        print(f"🔧 Token optimizer initialized for {model_name}")
        print(f"   Max context: {self.max_context_tokens} tokens")
        print(f"   Target response: {self.target_response_tokens} tokens")
    
    def _get_max_context_tokens(self) -> int:
        """Get maximum context tokens for the model"""
        context_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000
        }
        return context_limits.get(self.model_name, 4096)
    
    def _create_optimized_templates(self) -> Dict[str, str]:
        """Create prompt templates optimized for different token budgets"""
        return {
            "minimal": """UIUC CS advisor. Use context to answer concisely.
Context: {context}
Q: {question}
A:""",
            
            "compact": """You are a UIUC CS academic advisor. Answer using the context provided.

Context:
{context}

Question: {question}

Answer:""",
            
            "standard": """You are an expert UIUC academic advisor. Use the provided context to answer the student's question accurately.

Context Information:
{context}

Student Question: {question}

Guidelines:
1. Answer directly using the context
2. Include course numbers when relevant
3. Be concise but helpful

Response:""",
            
            "detailed": """You are an expert UIUC academic advisor with comprehensive knowledge of computer science courses, prerequisites, and student experiences.

Use the provided context to answer the student's question accurately and helpfully. Synthesize information from multiple sources when available.

Context Information:
{context}

Student Question: {question}

Guidelines for your response:
1. Provide direct, actionable answers
2. Include specific course numbers and names when relevant
3. Mention prerequisites when discussing courses
4. If discussing difficulty, provide context about what makes courses challenging
5. If you don't have enough information, say so clearly
6. Keep responses focused and well-structured

Response:"""
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def optimize_context_for_tokens(self, docs: List[Document], question: str, 
                                  max_context_tokens: Optional[int] = None) -> Tuple[str, Dict]:
        """Optimize context to fit within token budget"""
        if max_context_tokens is None:
            # Calculate available tokens for context
            question_tokens = self.count_tokens(question)
            template_tokens = self.count_tokens(self.prompt_templates["standard"].format(context="", question=""))
            max_context_tokens = (self.max_context_tokens - 
                                 self.target_response_tokens - 
                                 self.reserved_tokens - 
                                 question_tokens - 
                                 template_tokens)
        
        print(f"🎯 Optimizing context for {max_context_tokens} tokens")
        
        # Start with all documents and progressively optimize
        optimized_docs = self._prioritize_documents(docs, question)
        context_parts = []
        current_tokens = 0
        used_docs = 0
        
        for doc in optimized_docs:
            # Create context part
            content = doc.page_content
            course_codes = doc.metadata.get('course_codes', [])
            content_type = doc.metadata.get('content_type', 'general')
            
            # Try different context formats based on available space
            context_options = [
                # Minimal format
                f"{content}",
                # Compact format  
                f"[{content_type}] {content}",
                # Standard format
                f"Source {used_docs + 1} ({content_type}): {content}"
            ]
            
            # Find the format that fits
            best_context = None
            for context_option in context_options:
                tokens = self.count_tokens(context_option)
                if current_tokens + tokens <= max_context_tokens:
                    best_context = context_option
                    break
            
            if best_context:
                context_parts.append(best_context)
                current_tokens += self.count_tokens(best_context)
                used_docs += 1
            else:
                # Try to fit a truncated version
                truncated = self._truncate_to_fit(content, max_context_tokens - current_tokens - 20)
                if truncated:
                    context_parts.append(truncated)
                    current_tokens += self.count_tokens(truncated)
                    used_docs += 1
                break  # Can't fit more
        
        context = "\\n\\n".join(context_parts)
        
        optimization_stats = {
            'original_docs': len(docs),
            'used_docs': used_docs,
            'context_tokens': current_tokens,
            'max_context_tokens': max_context_tokens,
            'utilization': current_tokens / max_context_tokens if max_context_tokens > 0 else 0
        }
        
        print(f"📊 Context optimization: {used_docs}/{len(docs)} docs, {current_tokens}/{max_context_tokens} tokens ({optimization_stats['utilization']:.1%})")
        
        return context, optimization_stats
    
    def _prioritize_documents(self, docs: List[Document], question: str) -> List[Document]:
        """Prioritize documents based on relevance and information density"""
        question_words = set(question.lower().split())
        
        scored_docs = []
        for doc in docs:
            content = doc.page_content.lower()
            content_words = set(content.split())
            
            # Calculate relevance scores
            word_overlap = len(question_words.intersection(content_words))
            content_length = len(content.split())
            
            # Information density score
            density = word_overlap / content_length if content_length > 0 else 0
            
            # Content type priority
            content_type = doc.metadata.get('content_type', 'general')
            type_priority = {
                'prerequisites': 3,
                'course_description': 2,
                'student_advice': 1,
                'general': 0
            }.get(content_type, 0)
            
            # Combined score
            score = word_overlap + density * 10 + type_priority
            scored_docs.append((score, doc))
        
        # Sort by score and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]
    
    def _truncate_to_fit(self, text: str, max_tokens: int) -> Optional[str]:
        """Truncate text to fit within token limit"""
        if max_tokens <= 0:
            return None
        
        # Try to truncate intelligently at sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        truncated_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.count_tokens(sentence + ". ")
            if current_tokens + sentence_tokens <= max_tokens:
                truncated_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        if truncated_sentences:
            result = ". ".join(truncated_sentences) + "."
            # Add truncation indicator if we cut off content
            if len(truncated_sentences) < len([s for s in sentences if s.strip()]):
                result += " [...]"
            return result
        
        # If no complete sentences fit, truncate by words
        words = text.split()
        truncated_words = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens <= max_tokens - 10:  # Reserve for [...]
                truncated_words.append(word)
                current_tokens += word_tokens
            else:
                break
        
        if truncated_words:
            return " ".join(truncated_words) + " [...]"
        
        return None
    
    def select_optimal_prompt_template(self, question: str, context: str) -> Tuple[str, str]:
        """Select the best prompt template based on token budget"""
        question_tokens = self.count_tokens(question)
        context_tokens = self.count_tokens(context)
        
        # Calculate remaining budget
        used_tokens = question_tokens + context_tokens + self.target_response_tokens + self.reserved_tokens
        remaining_budget = self.max_context_tokens - used_tokens
        
        # Select template based on remaining budget
        if remaining_budget >= 200:
            template_key = "detailed"
        elif remaining_budget >= 100:
            template_key = "standard"
        elif remaining_budget >= 50:
            template_key = "compact"
        else:
            template_key = "minimal"
        
        template = self.prompt_templates[template_key]
        template_tokens = self.count_tokens(template.format(context="CONTEXT", question="QUESTION"))
        
        print(f"🎯 Selected '{template_key}' template ({template_tokens} tokens, {remaining_budget} budget remaining)")
        
        return template, template_key
    
    def optimize_full_prompt(self, docs: List[Document], question: str) -> Tuple[str, Dict]:
        """Optimize the entire prompt for token efficiency"""
        # First, optimize the context
        context, context_stats = self.optimize_context_for_tokens(docs, question)
        
        # Select optimal template
        template, template_key = self.select_optimal_prompt_template(question, context)
        
        # Create final prompt
        final_prompt = template.format(context=context, question=question)
        final_tokens = self.count_tokens(final_prompt)
        
        # Calculate total expected tokens
        total_expected = final_tokens + self.target_response_tokens
        
        optimization_stats = {
            **context_stats,
            'template_used': template_key,
            'prompt_tokens': final_tokens,
            'expected_response_tokens': self.target_response_tokens,
            'total_expected_tokens': total_expected,
            'budget_utilization': total_expected / self.max_context_tokens,
            'within_budget': total_expected <= self.max_context_tokens
        }
        
        print(f"📋 Final prompt: {final_tokens} tokens + {self.target_response_tokens} response = {total_expected} total")
        print(f"💰 Budget utilization: {optimization_stats['budget_utilization']:.1%}")
        
        if not optimization_stats['within_budget']:
            print("⚠️ WARNING: Prompt may exceed token budget!")
        
        return final_prompt, optimization_stats
    
    def estimate_api_cost(self, prompt_tokens: int, response_tokens: int, model: str = None) -> Dict:
        """Estimate API cost for the prompt"""
        model = model or self.model_name
        
        # Pricing per 1K tokens (as of 2024 - these may change)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
        
        if model not in pricing:
            model = "gpt-3.5-turbo"  # Default
        
        input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (response_tokens / 1000) * pricing[model]["output"]
        total_cost = input_cost + output_cost
        
        return {
            'model': model,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization capabilities"""
        return {
            'model': self.model_name,
            'max_context_tokens': self.max_context_tokens,
            'target_response_tokens': self.target_response_tokens,
            'reserved_tokens': self.reserved_tokens,
            'available_templates': list(self.prompt_templates.keys()),
            'encoding': str(self.encoding)
        }