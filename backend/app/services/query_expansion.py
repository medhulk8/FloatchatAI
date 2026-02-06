"""
query_expansion.py
Query expansion for better retrieval - adds synonyms and related terms
"""
from typing import Dict, Any, List, Set
import structlog
import re

logger = structlog.get_logger()


class QueryExpander:
    """Expands queries with synonyms and related terms for better retrieval"""

    def __init__(self):
        # Oceanographic domain-specific synonyms and related terms
        self.synonyms = {
            # Temperature terms
            'temperature': ['temp', 'thermal', 'heat', 'warmth', 'sst'],
            'cold': ['cool', 'low temperature', 'frigid', 'chilled'],
            'warm': ['hot', 'high temperature', 'heated'],

            # Salinity terms
            'salinity': ['salt', 'saltiness', 'salt content', 'sss'],
            'fresh': ['low salinity', 'freshwater', 'diluted'],
            'saline': ['salty', 'high salinity', 'brackish'],

            # Pressure/Depth terms
            'pressure': ['depth pressure', 'hydrostatic pressure'],
            'depth': ['deep', 'depth level', 'vertical position', 'water depth'],
            'surface': ['top', 'sea surface', 'uppermost'],
            'deep': ['bottom', 'deep water', 'abyssal', 'depth'],

            # Geographic terms
            'ocean': ['sea', 'marine', 'oceanic'],
            'indian ocean': ['indian sea', 'arabian sea', 'bay of bengal'],
            'pacific': ['pacific ocean'],
            'atlantic': ['atlantic ocean'],
            'region': ['area', 'zone', 'location', 'waters'],
            'equator': ['equatorial', 'tropics', 'tropical'],
            'polar': ['arctic', 'antarctic', 'southern ocean'],

            # Oceanographic features
            'current': ['circulation', 'flow', 'stream', 'ocean current'],
            'upwelling': ['upward flow', 'nutrient rise'],
            'thermocline': ['temperature gradient', 'thermal layer'],
            'halocline': ['salinity gradient', 'salinity layer'],
            'mixed layer': ['surface layer', 'mixing depth'],

            # BGC parameters
            'oxygen': ['o2', 'dissolved oxygen', 'do'],
            'ph': ['acidity', 'alkalinity', 'hydrogen ion'],
            'nitrate': ['no3', 'nitrogen', 'nutrients'],
            'chlorophyll': ['chl', 'chlorophyll-a', 'phytoplankton'],

            # Time-related
            'recent': ['latest', 'new', 'current', 'newest'],
            'trend': ['pattern', 'change over time', 'temporal variation'],
            'seasonal': ['monsoon', 'summer', 'winter', 'yearly cycle'],

            # Statistical terms
            'average': ['mean', 'avg', 'typical'],
            'maximum': ['max', 'highest', 'peak'],
            'minimum': ['min', 'lowest', 'bottom'],
            'range': ['variation', 'spread', 'difference'],

            # Data terms
            'profile': ['vertical profile', 'measurement', 'observation'],
            'float': ['argo float', 'profiling float', 'autonomous float'],
            'data': ['measurements', 'observations', 'values', 'readings']
        }

        # Contextual expansions (add context-specific terms)
        self.contextual_terms = {
            'temperature': ['stratification', 'warming', 'cooling', 'gradient'],
            'salinity': ['evaporation', 'precipitation', 'river discharge'],
            'ocean': ['marine ecosystem', 'circulation', 'water mass'],
            'climate': ['climate change', 'global warming', 'variability']
        }

        # Regional context
        self.regional_context = {
            'indian ocean': ['monsoon', 'arabian sea', 'bay of bengal', 'tropical'],
            'pacific': ['el niño', 'la niña', 'enso', 'kuroshio'],
            'atlantic': ['gulf stream', 'north atlantic', 'amoc'],
            'southern ocean': ['antarctic', 'polar', 'circumpolar current']
        }

    def expand_query(
        self,
        query: str,
        intent_result: Dict[str, Any] = None,
        max_expansions: int = 5
    ) -> Dict[str, Any]:
        """
        Expand query with synonyms and related terms

        Returns:
            {
                "original_query": str,
                "expanded_query": str,
                "expansion_terms": List[str],
                "expansion_applied": bool
            }
        """

        query_lower = query.lower().strip()

        # Extract key terms from query
        key_terms = self._extract_key_terms(query_lower)

        if not key_terms:
            return {
                "original_query": query,
                "expanded_query": query,
                "expansion_terms": [],
                "expansion_applied": False
            }

        # Find synonyms for key terms
        expansion_terms = []
        for term in key_terms:
            synonyms = self._get_synonyms(term)
            if synonyms:
                expansion_terms.extend(synonyms[:2])  # Add top 2 synonyms per term

        # Limit total expansions
        expansion_terms = list(set(expansion_terms))[:max_expansions]

        # Build expanded query
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"

            logger.info(
                "Query expanded",
                original=query,
                expansions=expansion_terms,
                count=len(expansion_terms)
            )

            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "expansion_terms": expansion_terms,
                "expansion_applied": True
            }
        else:
            return {
                "original_query": query,
                "expanded_query": query,
                "expansion_terms": [],
                "expansion_applied": False
            }

    def expand_for_vector_search(
        self,
        query: str,
        intent_result: Dict[str, Any] = None
    ) -> str:
        """
        Expand query specifically for vector search (more aggressive)

        Vector search benefits from richer context
        """

        expansion_result = self.expand_query(query, intent_result, max_expansions=8)

        # Add contextual terms based on intent
        if intent_result:
            primary_intent = intent_result.get('primary_intent', '')

            if 'pattern' in primary_intent or 'analysis' in primary_intent:
                # Add analytical context
                expansion_result['expanded_query'] += " trends patterns behavior characteristics"

            elif 'temporal' in primary_intent:
                # Add temporal context
                expansion_result['expanded_query'] += " time series changes evolution"

            elif 'spatial' in primary_intent:
                # Add spatial context
                expansion_result['expanded_query'] += " geographic regional location distribution"

        return expansion_result['expanded_query']

    def expand_for_sql_search(
        self,
        query: str,
        intent_result: Dict[str, Any] = None
    ) -> str:
        """
        Expand query for SQL search (conservative - only key synonyms)

        SQL search needs precise terms
        """

        expansion_result = self.expand_query(query, intent_result, max_expansions=3)
        return expansion_result['expanded_query']

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key oceanographic terms from query"""

        key_terms = []

        # Check for multi-word terms first
        for term in sorted(self.synonyms.keys(), key=len, reverse=True):
            if term in query and ' ' in term:  # Multi-word terms
                key_terms.append(term)

        # Check for single-word terms
        words = re.findall(r'\b\w+\b', query)
        for word in words:
            if word in self.synonyms and word not in key_terms:
                key_terms.append(word)

        return key_terms

    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term"""

        synonyms = []

        # Direct synonyms
        if term in self.synonyms:
            synonyms.extend(self.synonyms[term])

        # Contextual terms
        if term in self.contextual_terms:
            synonyms.extend(self.contextual_terms[term])

        # Regional context
        if term in self.regional_context:
            synonyms.extend(self.regional_context[term])

        return list(set(synonyms))  # Remove duplicates

    def get_query_variants(self, query: str, n: int = 3) -> List[str]:
        """
        Generate multiple query variants for ensemble retrieval

        Useful for comprehensive search
        """

        variants = [query]  # Original query

        # Variant 1: With synonyms
        expansion_result = self.expand_query(query, max_expansions=3)
        if expansion_result['expansion_applied']:
            variants.append(expansion_result['expanded_query'])

        # Variant 2: Focus on parameters
        param_focused = self._create_parameter_focused_variant(query)
        if param_focused != query:
            variants.append(param_focused)

        # Variant 3: Focus on location
        location_focused = self._create_location_focused_variant(query)
        if location_focused != query:
            variants.append(location_focused)

        return variants[:n]

    def _create_parameter_focused_variant(self, query: str) -> str:
        """Create variant focusing on oceanographic parameters"""

        params = ['temperature', 'salinity', 'pressure', 'oxygen', 'ph', 'nitrate']
        query_params = [p for p in params if p in query.lower()]

        if query_params:
            return f"{query} {' '.join([p + ' data' for p in query_params])}"

        return query

    def _create_location_focused_variant(self, query: str) -> str:
        """Create variant focusing on geographic location"""

        regions = ['indian ocean', 'pacific', 'atlantic', 'southern ocean',
                   'arabian sea', 'bay of bengal']
        query_regions = [r for r in regions if r in query.lower()]

        if query_regions:
            return f"{query} region location coordinates"

        return query


# Global instance
query_expander = QueryExpander()
