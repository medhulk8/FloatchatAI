"""
query_intent_detector.py
Advanced query intent detection for better query routing and processing
"""
from typing import Dict, Any, List, Tuple
from enum import Enum
import re
import structlog

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    """Specific query intent types"""
    # Data Retrieval Intents
    DATA_RETRIEVAL = "data_retrieval"  # "Show me", "Find", "Get"
    STATISTICAL = "statistical"  # "Average", "Mean", "Count"
    COMPARISON = "comparison"  # "Compare X vs Y"
    TEMPORAL = "temporal"  # Time-series, trends over time
    SPATIAL = "spatial"  # Geographic, regional queries

    # Analytical Intents
    PATTERN_ANALYSIS = "pattern_analysis"  # "Patterns", "Trends"
    EXPLANATION = "explanation"  # "Why", "How", "Explain"
    DESCRIPTION = "description"  # "Describe", "What is"

    # Visualization Intents
    VISUALIZATION = "visualization"  # "Create chart", "Show map"
    TABLE_VIEW = "table_view"  # "Table", "List view"

    # Meta Intents
    CAPABILITY = "capability"  # "What can you do?"
    HELP = "help"  # "Help me"


class QueryIntentDetector:
    """Detects granular query intent for better processing"""

    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.STATISTICAL: {
                'keywords': [
                    'average', 'mean', 'median', 'sum', 'total', 'count',
                    'maximum', 'minimum', 'std', 'standard deviation',
                    'statistics', 'stats', 'aggregate'
                ],
                'patterns': [
                    r'what is the (average|mean|median)',
                    r'calculate (average|mean|sum|total)',
                    r'how many',
                ]
            },
            QueryIntent.COMPARISON: {
                'keywords': [
                    'compare', 'comparison', 'versus', 'vs', 'difference',
                    'between', 'contrast', 'relative to'
                ],
                'patterns': [
                    r'compare (\w+) (and|vs|versus) (\w+)',
                    r'(\w+) vs (\w+)',
                    r'difference between (\w+) and (\w+)',
                ]
            },
            QueryIntent.TEMPORAL: {
                'keywords': [
                    'trend', 'over time', 'time series', 'temporal',
                    'change over', 'evolution', 'progression', 'history',
                    'yearly', 'monthly', 'seasonal', 'annual'
                ],
                'patterns': [
                    r'(trend|change) over (time|years|months)',
                    r'from \d{4} to \d{4}',
                    r'between \d{4} and \d{4}',
                ]
            },
            QueryIntent.SPATIAL: {
                'keywords': [
                    'near', 'around', 'region', 'area', 'location',
                    'geographic', 'spatial', 'coordinates', 'latitude',
                    'longitude', 'ocean', 'sea', 'bay'
                ],
                'patterns': [
                    r'near (\d+\.?\d*)°?\s*[NS],?\s*(\d+\.?\d*)°?\s*[EW]',
                    r'in (the)? \w+ (ocean|sea|bay)',
                    r'around coordinates',
                ]
            },
            QueryIntent.PATTERN_ANALYSIS: {
                'keywords': [
                    'pattern', 'trend', 'behavior', 'characteristic',
                    'typical', 'common', 'usual', 'generally',
                    'insight', 'analysis', 'understand'
                ],
                'patterns': [
                    r'what are the (patterns|trends|characteristics)',
                    r'analyze (the)? \w+',
                ]
            },
            QueryIntent.EXPLANATION: {
                'keywords': [
                    'why', 'how', 'explain', 'reason', 'cause',
                    'what causes', 'mechanism', 'process'
                ],
                'patterns': [
                    r'^(why|how) ',
                    r'explain (why|how)',
                ]
            },
            QueryIntent.DESCRIPTION: {
                'keywords': [
                    'describe', 'what is', 'what are', 'tell me about',
                    'information about', 'details about'
                ],
                'patterns': [
                    r'describe (the)? \w+',
                    r'what (is|are) \w+',
                ]
            },
            QueryIntent.VISUALIZATION: {
                'keywords': [
                    'chart', 'graph', 'plot', 'visualize', 'visualization',
                    'map', 'bar chart', 'line graph', 'scatter plot'
                ],
                'patterns': [
                    r'(create|show|generate) (a|an)? (chart|graph|map|plot)',
                    r'visualize \w+',
                ]
            },
            QueryIntent.TABLE_VIEW: {
                'keywords': [
                    'table', 'tabular', 'list', 'data table',
                    'spreadsheet', 'grid'
                ],
                'patterns': [
                    r'(show|display|create) (a|an)? table',
                    r'in tabular form',
                ]
            },
            QueryIntent.DATA_RETRIEVAL: {
                'keywords': [
                    'show', 'find', 'get', 'fetch', 'retrieve', 'display',
                    'give me', 'provide', 'list', 'search'
                ],
                'patterns': [
                    r'^(show|find|get|fetch|display)',
                    r'give me (the)? \w+',
                ]
            },
        }

    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect query intent with confidence scores

        Returns:
            {
                "primary_intent": QueryIntent,
                "secondary_intents": List[QueryIntent],
                "confidence": float,
                "suggested_retrieval": str,  # "sql", "vector", "hybrid"
                "analysis": Dict
            }
        """
        query_lower = query.lower().strip()

        # Check for capability/help queries first
        if self._is_capability_query(query_lower):
            return self._create_intent_result(
                QueryIntent.CAPABILITY,
                confidence=1.0,
                retrieval="none"
            )

        # Score all intents
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_intent_score(query_lower, patterns)
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            # Default to data retrieval
            return self._create_intent_result(
                QueryIntent.DATA_RETRIEVAL,
                confidence=0.5,
                retrieval="sql"
            )

        # Sort by score
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_intent = sorted_intents[0][0]
        primary_confidence = sorted_intents[0][1]

        # Get secondary intents (score > 0.3)
        secondary_intents = [
            intent for intent, score in sorted_intents[1:]
            if score > 0.3
        ]

        # Determine retrieval strategy
        retrieval_strategy = self._determine_retrieval_strategy(
            primary_intent,
            secondary_intents
        )

        # Additional analysis
        analysis = {
            "all_scores": {str(k): v for k, v in intent_scores.items()},
            "query_complexity": self._assess_query_complexity(query),
            "requires_computation": self._requires_computation(primary_intent),
            "requires_context": self._requires_context(primary_intent)
        }

        return {
            "primary_intent": primary_intent,
            "secondary_intents": secondary_intents,
            "confidence": primary_confidence,
            "suggested_retrieval": retrieval_strategy,
            "analysis": analysis
        }

    def _calculate_intent_score(
        self,
        query: str,
        patterns: Dict[str, List[str]]
    ) -> float:
        """Calculate score for a specific intent"""
        score = 0.0

        # Keyword matching
        keywords = patterns.get('keywords', [])
        keyword_matches = sum(1 for kw in keywords if kw in query)
        if keyword_matches > 0:
            score += min(keyword_matches * 0.3, 0.6)  # Max 0.6 from keywords

        # Pattern matching
        regex_patterns = patterns.get('patterns', [])
        for pattern in regex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.4  # Higher weight for pattern matches
                break  # Only count one pattern match

        return min(score, 1.0)

    def _is_capability_query(self, query: str) -> bool:
        """Check if query is asking about capabilities"""
        capability_phrases = [
            'what can you do', 'what are your capabilities',
            'help', 'what features', 'capabilities',
            'who are you', 'what are you'
        ]
        return any(phrase in query for phrase in capability_phrases)

    def _determine_retrieval_strategy(
        self,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent]
    ) -> str:
        """Determine optimal retrieval strategy based on intents"""

        # Vector retrieval intents (conceptual, analytical)
        vector_intents = {
            QueryIntent.PATTERN_ANALYSIS,
            QueryIntent.EXPLANATION,
            QueryIntent.DESCRIPTION
        }

        # SQL retrieval intents (precise data)
        sql_intents = {
            QueryIntent.DATA_RETRIEVAL,
            QueryIntent.STATISTICAL,
            QueryIntent.COMPARISON,
            QueryIntent.TABLE_VIEW
        }

        # Hybrid intents (needs both)
        hybrid_intents = {
            QueryIntent.TEMPORAL,
            QueryIntent.SPATIAL,
            QueryIntent.VISUALIZATION
        }

        # Check primary intent
        if primary_intent in vector_intents:
            return "vector"
        elif primary_intent in sql_intents:
            return "sql"
        elif primary_intent in hybrid_intents:
            return "hybrid"

        # Check if secondary intents suggest hybrid
        has_vector_secondary = any(i in vector_intents for i in secondary_intents)
        has_sql_secondary = any(i in sql_intents for i in secondary_intents)

        if has_vector_secondary and has_sql_secondary:
            return "hybrid"

        # Default based on primary
        return "sql" if primary_intent in sql_intents else "vector"

    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of the query"""
        word_count = len(query.split())

        # Check for complex patterns
        has_and = ' and ' in query.lower()
        has_or = ' or ' in query.lower()
        has_multiple_conditions = bool(re.search(r'(where|when|if|between)', query, re.I))

        if word_count > 15 or (has_and and has_or) or has_multiple_conditions:
            return "complex"
        elif word_count > 8 or has_and or has_or:
            return "moderate"
        else:
            return "simple"

    def _requires_computation(self, intent: QueryIntent) -> bool:
        """Check if intent requires computation"""
        computational_intents = {
            QueryIntent.STATISTICAL,
            QueryIntent.COMPARISON,
            QueryIntent.TEMPORAL
        }
        return intent in computational_intents

    def _requires_context(self, intent: QueryIntent) -> bool:
        """Check if intent requires contextual understanding"""
        contextual_intents = {
            QueryIntent.PATTERN_ANALYSIS,
            QueryIntent.EXPLANATION,
            QueryIntent.DESCRIPTION
        }
        return intent in contextual_intents

    def _create_intent_result(
        self,
        intent: QueryIntent,
        confidence: float,
        retrieval: str,
        secondary: List[QueryIntent] = None
    ) -> Dict[str, Any]:
        """Helper to create intent result"""
        return {
            "primary_intent": intent,
            "secondary_intents": secondary or [],
            "confidence": confidence,
            "suggested_retrieval": retrieval,
            "analysis": {
                "query_complexity": "simple",
                "requires_computation": False,
                "requires_context": False
            }
        }


# Global query intent detector instance
query_intent_detector = QueryIntentDetector()
