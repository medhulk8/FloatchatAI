"""
result_ranker.py
Intelligent ranking of hybrid retrieval results combining SQL and vector search
"""
from typing import List, Dict, Any, Tuple
import structlog
from datetime import datetime
import math

logger = structlog.get_logger()


class ResultRanker:
    """Ranks and combines SQL and vector search results intelligently"""

    def __init__(self):
        # Configurable weights for hybrid scoring
        self.sql_weight = 0.6  # Higher weight for precise SQL results
        self.vector_weight = 0.4  # Lower weight for semantic similarity
        self.recency_boost = 0.1  # Boost for recent data
        self.geographic_boost = 0.15  # Boost for geographic relevance

    def rank_hybrid_results(
        self,
        sql_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        query: str,
        entities: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine and rank SQL and vector results using multi-factor scoring

        Args:
            sql_results: Results from SQL retrieval
            vector_results: Results from vector search
            query: Original user query
            entities: Extracted entities (regions, parameters, dates)

        Returns:
            Ranked list of results with scores
        """
        entities = entities or {}

        logger.info(
            "Ranking hybrid results",
            sql_count=len(sql_results),
            vector_count=len(vector_results)
        )

        # Score and normalize SQL results
        scored_sql = self._score_sql_results(sql_results, query, entities)

        # Score and normalize vector results
        scored_vector = self._score_vector_results(vector_results, query, entities)

        # Combine results
        combined_results = self._combine_results(scored_sql, scored_vector)

        # Re-rank with cross-validation
        final_ranked = self._rerank_with_cross_validation(
            combined_results, query, entities
        )

        logger.info(
            "Hybrid ranking completed",
            total_results=len(final_ranked),
            top_score=final_ranked[0]['final_score'] if final_ranked else 0
        )

        return final_ranked

    def _score_sql_results(
        self,
        sql_results: List[Dict[str, Any]],
        query: str,
        entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score SQL results based on relevance factors"""

        scored_results = []

        for idx, result in enumerate(sql_results):
            score = 1.0  # Base score for SQL (highly relevant)

            # Position-based decay (earlier results slightly preferred)
            position_score = 1.0 - (idx * 0.01)  # Decay by 1% per position
            score *= position_score

            # Recency boost if date available
            if 'profile_date' in result or 'date' in result:
                date_field = result.get('profile_date') or result.get('date')
                recency_score = self._calculate_recency_score(date_field)
                score += recency_score * self.recency_boost

            # Geographic relevance boost
            if entities.get('regions') and ('region' in result or 'latitude' in result):
                geo_score = self._calculate_geographic_relevance(result, entities['regions'])
                score += geo_score * self.geographic_boost

            # Parameter availability boost
            if entities.get('parameters'):
                param_score = self._calculate_parameter_score(result, entities['parameters'])
                score += param_score * 0.1

            scored_results.append({
                **result,
                'source': 'sql',
                'base_score': score,
                'normalized_score': score * self.sql_weight
            })

        return scored_results

    def _score_vector_results(
        self,
        vector_results: List[Dict[str, Any]],
        query: str,
        entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score vector results based on similarity and relevance"""

        scored_results = []

        # Get similarity scores (distance in ChromaDB - lower is better)
        max_distance = max(
            (r.get('distance', 1.0) for r in vector_results),
            default=1.0
        )

        for result in vector_results:
            # Convert distance to similarity (1 - normalized_distance)
            distance = result.get('distance', 1.0)
            similarity = 1.0 - (distance / max_distance if max_distance > 0 else 0)

            score = similarity  # Base score from vector similarity

            # Metadata quality boost
            metadata = result.get('metadata', {})
            if self._has_complete_metadata(metadata):
                score += 0.1

            # Geographic relevance
            if entities.get('regions') and 'region' in metadata:
                if metadata['region'] in [r.lower() for r in entities['regions']]:
                    score += self.geographic_boost

            # Recency boost
            if 'date' in metadata:
                recency_score = self._calculate_recency_score(metadata['date'])
                score += recency_score * self.recency_boost

            scored_results.append({
                **result,
                'source': 'vector',
                'base_score': score,
                'normalized_score': score * self.vector_weight
            })

        return scored_results

    def _combine_results(
        self,
        scored_sql: List[Dict[str, Any]],
        scored_vector: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine SQL and vector results, removing duplicates"""

        combined = []
        seen_ids = set()

        # Prioritize SQL results (more precise)
        for result in scored_sql:
            result_id = self._get_result_id(result)
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                combined.append(result)

        # Add unique vector results
        for result in scored_vector:
            result_id = self._get_result_id(result)
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                combined.append(result)

        return combined

    def _rerank_with_cross_validation(
        self,
        combined_results: List[Dict[str, Any]],
        query: str,
        entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Final re-ranking with cross-validation between sources"""

        # If result appears in both SQL and vector, boost its score
        sql_ids = {self._get_result_id(r) for r in combined_results if r.get('source') == 'sql'}
        vector_ids = {self._get_result_id(r) for r in combined_results if r.get('source') == 'vector'}
        cross_validated_ids = sql_ids & vector_ids

        for result in combined_results:
            result_id = self._get_result_id(result)

            # Cross-validation boost
            if result_id in cross_validated_ids:
                result['normalized_score'] *= 1.2  # 20% boost for cross-validation

            # Calculate final score
            result['final_score'] = result['normalized_score']

        # Sort by final score (descending)
        ranked = sorted(
            combined_results,
            key=lambda x: x['final_score'],
            reverse=True
        )

        return ranked

    def _calculate_recency_score(self, date_field: Any) -> float:
        """Calculate recency score (0.0 to 1.0) based on date"""
        try:
            if isinstance(date_field, str):
                date = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
            elif isinstance(date_field, datetime):
                date = date_field
            else:
                return 0.0

            # Calculate age in days
            age_days = (datetime.now() - date.replace(tzinfo=None)).days

            # Exponential decay: newer data gets higher score
            # Score = e^(-age/365), so 1-year-old data = ~0.37 score
            recency_score = math.exp(-age_days / 365.0)

            return recency_score

        except Exception as e:
            logger.debug("Failed to calculate recency score", error=str(e))
            return 0.0

    def _calculate_geographic_relevance(
        self,
        result: Dict[str, Any],
        requested_regions: List[str]
    ) -> float:
        """Calculate geographic relevance score"""
        try:
            # Direct region match
            if 'region' in result:
                result_region = str(result['region']).lower()
                for requested in requested_regions:
                    if requested.lower() in result_region or result_region in requested.lower():
                        return 1.0

            # Coordinate-based relevance (if applicable)
            # Could implement bounding box checks here

            return 0.0

        except Exception as e:
            logger.debug("Failed to calculate geographic relevance", error=str(e))
            return 0.0

    def _calculate_parameter_score(
        self,
        result: Dict[str, Any],
        requested_params: List[str]
    ) -> float:
        """Calculate score based on parameter availability"""
        try:
            available_params = [
                key for key in result.keys()
                if key in ['temperature', 'salinity', 'pressure', 'oxygen', 'ph', 'nitrate', 'chlorophyll']
            ]

            if not requested_params:
                return 0.0

            # Calculate overlap ratio
            matches = sum(1 for param in requested_params if param.lower() in available_params)
            score = matches / len(requested_params)

            return score

        except Exception as e:
            logger.debug("Failed to calculate parameter score", error=str(e))
            return 0.0

    def _has_complete_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata is complete (has key fields)"""
        required_fields = ['latitude', 'longitude', 'date']
        return all(field in metadata for field in required_fields)

    def _get_result_id(self, result: Dict[str, Any]) -> str:
        """Extract unique identifier from result"""
        # Try various ID fields
        for id_field in ['id', 'profile_id', 'float_id', 'record_id']:
            if id_field in result:
                return str(result[id_field])

        # Fallback: use metadata ID
        metadata = result.get('metadata', {})
        for id_field in ['id', 'profile_id', 'float_id']:
            if id_field in metadata:
                return str(metadata[id_field])

        # Last resort: generate from coordinates + date
        lat = result.get('latitude') or metadata.get('latitude', '')
        lon = result.get('longitude') or metadata.get('longitude', '')
        date = result.get('date') or metadata.get('date', '')
        return f"{lat}_{lon}_{date}"


# Global result ranker instance
result_ranker = ResultRanker()
