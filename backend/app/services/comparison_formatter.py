"""
comparison_formatter.py
Intelligent comparison formatter for comparative queries - generates markdown tables
"""
from typing import Dict, Any, List, Tuple, Optional
import structlog
from datetime import datetime
from collections import defaultdict
import re

logger = structlog.get_logger()


class ComparisonFormatter:
    """Formats comparison query results as structured tables"""

    def __init__(self):
        self.month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

    def should_use_comparison_format(self, query: str, intent_result: Dict[str, Any]) -> bool:
        """Determine if query should use comparison table format"""

        # Check intent
        if intent_result.get('primary_intent') == 'comparison':
            return True

        # Check for comparison keywords
        comparison_keywords = [
            'compare', 'comparison', 'versus', 'vs', 'difference between',
            'compared to', 'relative to', 'vs.', 'contrast'
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in comparison_keywords)

    def format_comparison(
        self,
        query: str,
        sql_results: List[Dict[str, Any]],
        intent_result: Dict[str, Any]
    ) -> str:
        """
        Generate comparison table from SQL results

        Automatically detects comparison dimension:
        - Temporal: months, years, seasons
        - Spatial: regions, locations
        - Parametric: temperature vs salinity
        """

        if not sql_results:
            return "No data available for comparison."

        logger.info("Formatting comparison query", query=query, result_count=len(sql_results))

        # Detect comparison dimension
        comparison_type = self._detect_comparison_type(query, sql_results)

        if comparison_type == 'temporal_month':
            return self._format_temporal_month_comparison(query, sql_results)
        elif comparison_type == 'temporal_year':
            return self._format_temporal_year_comparison(query, sql_results)
        elif comparison_type == 'spatial_region':
            return self._format_spatial_comparison(query, sql_results)
        elif comparison_type == 'parametric':
            return self._format_parametric_comparison(query, sql_results)
        else:
            # Generic comparison
            return self._format_generic_comparison(query, sql_results)

    def _detect_comparison_type(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Detect what dimension is being compared"""

        query_lower = query.lower()

        # Check for years first (4-digit numbers)
        year_matches = re.findall(r'\b(?:19|20)\d{2}\b', query)
        has_multiple_years = len(set(year_matches)) >= 2

        # Check for month names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        has_month = any(month in query_lower for month in months)

        # PRIORITY 1: If query has both month names AND multiple years,
        # treat as year comparison (e.g., "july 2022 vs july 2023")
        if has_month and has_multiple_years:
            return 'temporal_year'

        # PRIORITY 2: If query has month names but NOT multiple years,
        # treat as month comparison (e.g., "july vs august")
        if has_month:
            return 'temporal_month'

        # PRIORITY 3: If query has multiple years, treat as year comparison
        if has_multiple_years:
            return 'temporal_year'

        # Check for regions
        regions = ['indian ocean', 'pacific', 'atlantic', 'arabian sea',
                   'bay of bengal', 'mediterranean', 'southern ocean']
        if any(region in query_lower for region in regions):
            return 'spatial_region'

        # Check for parameter comparison
        params = ['temperature', 'salinity', 'pressure', 'oxygen', 'ph']
        param_count = sum(1 for p in params if p in query_lower)
        if param_count >= 2:
            return 'parametric'

        return 'generic'

    def _format_temporal_month_comparison(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format month-to-month comparison as table"""

        # Group by month
        monthly_data = defaultdict(lambda: {
            'profiles': [],
            'temperatures': [],
            'salinities': [],
            'pressures': [],
            'locations': []
        })

        for row in results:
            # Extract month from profile_date
            date_field = row.get('profile_date') or row.get('date')
            if date_field:
                date_obj = None
                if isinstance(date_field, str):
                    try:
                        date_obj = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                    except:
                        continue
                elif isinstance(date_field, datetime):
                    date_obj = date_field
                elif hasattr(date_field, 'month') and hasattr(date_field, 'year'):
                    # Handle Python date objects (not datetime)
                    date_obj = date_field
                else:
                    continue

                if date_obj is None:
                    continue

                month = date_obj.month
                month_name = self.month_names[month]

                monthly_data[month_name]['profiles'].append(row.get('profile_id', ''))

                # Collect data
                if 'temperature' in row or 'surface_temperature' in row:
                    temp = row.get('surface_temperature') or (
                        row['temperature'][0] if isinstance(row.get('temperature'), list) else None
                    )
                    if temp is not None:
                        monthly_data[month_name]['temperatures'].append(float(temp))

                if 'salinity' in row or 'surface_salinity' in row:
                    sal = row.get('surface_salinity') or (
                        row['salinity'][0] if isinstance(row.get('salinity'), list) else None
                    )
                    if sal is not None:
                        monthly_data[month_name]['salinities'].append(float(sal))

                if 'latitude' in row and 'longitude' in row:
                    monthly_data[month_name]['locations'].append((
                        float(row['latitude']),
                        float(row['longitude'])
                    ))

        if not monthly_data:
            return "No temporal data available for comparison."

        # Build comparison table
        response_parts = []
        response_parts.append("## 📊 **Monthly Comparison**\n")

        # Create table header
        months_found = sorted(monthly_data.keys(), key=lambda x: list(self.month_names.values()).index(x))

        table_lines = []
        table_lines.append("| Metric | " + " | ".join(months_found) + " |")
        table_lines.append("|--------|" + "|".join(["--------"] * len(months_found)) + "|")

        # Profiles count
        profile_counts = [str(len(monthly_data[m]['profiles'])) for m in months_found]
        table_lines.append("| **Profiles** | " + " | ".join(profile_counts) + " |")

        # Average temperature
        temp_avgs = []
        for month in months_found:
            temps = monthly_data[month]['temperatures']
            if temps:
                avg = sum(temps) / len(temps)
                temp_avgs.append(f"{avg:.2f}°C")
            else:
                temp_avgs.append("N/A")
        table_lines.append("| **Avg Temperature** | " + " | ".join(temp_avgs) + " |")

        # Temperature range
        temp_ranges = []
        for month in months_found:
            temps = monthly_data[month]['temperatures']
            if temps:
                temp_ranges.append(f"{min(temps):.1f} - {max(temps):.1f}°C")
            else:
                temp_ranges.append("N/A")
        table_lines.append("| **Temp Range** | " + " | ".join(temp_ranges) + " |")

        # Average salinity
        sal_avgs = []
        for month in months_found:
            sals = monthly_data[month]['salinities']
            if sals:
                avg = sum(sals) / len(sals)
                sal_avgs.append(f"{avg:.2f} PSU")
            else:
                sal_avgs.append("N/A")
        table_lines.append("| **Avg Salinity** | " + " | ".join(sal_avgs) + " |")

        # Salinity range
        sal_ranges = []
        for month in months_found:
            sals = monthly_data[month]['salinities']
            if sals:
                sal_ranges.append(f"{min(sals):.1f} - {max(sals):.1f} PSU")
            else:
                sal_ranges.append("N/A")
        table_lines.append("| **Sal Range** | " + " | ".join(sal_ranges) + " |")

        # Geographic coverage
        geo_coverage = []
        for month in months_found:
            locs = monthly_data[month]['locations']
            if locs:
                lats = [loc[0] for loc in locs]
                lons = [loc[1] for loc in locs]
                geo_coverage.append(f"{len(locs)} locations")
            else:
                geo_coverage.append("N/A")
        table_lines.append("| **Coverage** | " + " | ".join(geo_coverage) + " |")

        response_parts.append("\n".join(table_lines))

        # Add summary
        response_parts.append(f"\n**Total Profiles Compared:** {sum(len(monthly_data[m]['profiles']) for m in months_found)}")

        # Calculate differences
        if len(months_found) == 2:
            m1, m2 = months_found[0], months_found[1]
            temps1 = monthly_data[m1]['temperatures']
            temps2 = monthly_data[m2]['temperatures']

            if temps1 and temps2:
                avg1 = sum(temps1) / len(temps1)
                avg2 = sum(temps2) / len(temps2)
                diff = avg2 - avg1

                response_parts.append(f"\n### 📈 **Key Difference**")
                response_parts.append(f"Temperature difference: **{abs(diff):.2f}°C** ({'warmer' if diff > 0 else 'cooler'} in {m2})")

            sals1 = monthly_data[m1]['salinities']
            sals2 = monthly_data[m2]['salinities']

            if sals1 and sals2:
                avg1 = sum(sals1) / len(sals1)
                avg2 = sum(sals2) / len(sals2)
                diff = avg2 - avg1

                response_parts.append(f"Salinity difference: **{abs(diff):.2f} PSU** ({'higher' if diff > 0 else 'lower'} in {m2})")

        return "\n".join(response_parts)

    def _format_temporal_year_comparison(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format year-to-year comparison as table"""

        logger.info("Starting year comparison formatting", result_count=len(results))

        # Group by year
        yearly_data = defaultdict(lambda: {
            'profiles': [],
            'temperatures': [],
            'salinities': []
        })

        for row in results:
            # Extract year - try the 'year' column first (from SQL EXTRACT)
            year = None

            if 'year' in row and row['year'] is not None:
                # Year column from SQL (may be Decimal or int)
                try:
                    year = int(row['year'])
                except (ValueError, TypeError):
                    pass

            # Fallback: extract from date field
            if year is None:
                date_field = row.get('profile_date') or row.get('date')
                if date_field:
                    if isinstance(date_field, str):
                        try:
                            date_obj = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                            year = date_obj.year
                        except:
                            continue
                    elif isinstance(date_field, datetime):
                        year = date_field.year
                    elif hasattr(date_field, 'year'):
                        # Handle Python date objects (not datetime)
                        year = date_field.year
                    else:
                        continue

            if year is None:
                continue

            yearly_data[year]['profiles'].append(row.get('profile_id', ''))

            if 'temperature' in row or 'surface_temperature' in row:
                temp = row.get('surface_temperature') or (
                    row['temperature'][0] if isinstance(row.get('temperature'), list) else None
                )
                if temp is not None:
                    yearly_data[year]['temperatures'].append(float(temp))

            if 'salinity' in row or 'surface_salinity' in row:
                sal = row.get('surface_salinity') or (
                    row['salinity'][0] if isinstance(row.get('salinity'), list) else None
                )
                if sal is not None:
                    yearly_data[year]['salinities'].append(float(sal))

        if not yearly_data:
            return "No temporal data available for comparison."

        # Build table
        response_parts = []
        response_parts.append("## 📊 **Year-to-Year Comparison**\n")

        years_found = sorted(yearly_data.keys())

        table_lines = []
        table_lines.append("| Metric | " + " | ".join(str(y) for y in years_found) + " |")
        table_lines.append("|--------|" + "|".join(["--------"] * len(years_found)) + "|")

        # Profiles
        profile_counts = [str(len(yearly_data[y]['profiles'])) for y in years_found]
        table_lines.append("| **Profiles** | " + " | ".join(profile_counts) + " |")

        # Temperature
        temp_avgs = []
        for year in years_found:
            temps = yearly_data[year]['temperatures']
            if temps:
                temp_avgs.append(f"{sum(temps)/len(temps):.2f}°C")
            else:
                temp_avgs.append("N/A")
        table_lines.append("| **Avg Temp** | " + " | ".join(temp_avgs) + " |")

        # Salinity
        sal_avgs = []
        for year in years_found:
            sals = yearly_data[year]['salinities']
            if sals:
                sal_avgs.append(f"{sum(sals)/len(sals):.2f} PSU")
            else:
                sal_avgs.append("N/A")
        table_lines.append("| **Avg Salinity** | " + " | ".join(sal_avgs) + " |")

        response_parts.append("\n".join(table_lines))

        return "\n".join(response_parts)

    def _format_spatial_comparison(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format region-to-region comparison"""

        # Group by region (if available) or by lat/lon zones
        regional_data = defaultdict(lambda: {
            'profiles': [],
            'temperatures': [],
            'salinities': []
        })

        for row in results:
            region = row.get('region', 'Unknown')

            # If no region, infer from coordinates
            if region == 'Unknown' and 'latitude' in row and 'longitude' in row:
                lat, lon = float(row['latitude']), float(row['longitude'])
                region = self._infer_region(lat, lon)

            regional_data[region]['profiles'].append(row.get('profile_id', ''))

            if 'temperature' in row or 'surface_temperature' in row:
                temp = row.get('surface_temperature') or (
                    row['temperature'][0] if isinstance(row.get('temperature'), list) else None
                )
                if temp is not None:
                    regional_data[region]['temperatures'].append(float(temp))

            if 'salinity' in row or 'surface_salinity' in row:
                sal = row.get('surface_salinity') or (
                    row['salinity'][0] if isinstance(row.get('salinity'), list) else None
                )
                if sal is not None:
                    regional_data[region]['salinities'].append(float(sal))

        # Build table
        response_parts = []
        response_parts.append("## 📊 **Regional Comparison**\n")

        regions_found = sorted(regional_data.keys())

        table_lines = []
        table_lines.append("| Metric | " + " | ".join(regions_found) + " |")
        table_lines.append("|--------|" + "|".join(["--------"] * len(regions_found)) + "|")

        # Profiles
        profile_counts = [str(len(regional_data[r]['profiles'])) for r in regions_found]
        table_lines.append("| **Profiles** | " + " | ".join(profile_counts) + " |")

        # Temperature
        temp_avgs = []
        for region in regions_found:
            temps = regional_data[region]['temperatures']
            if temps:
                temp_avgs.append(f"{sum(temps)/len(temps):.2f}°C")
            else:
                temp_avgs.append("N/A")
        table_lines.append("| **Avg Temp** | " + " | ".join(temp_avgs) + " |")

        # Salinity
        sal_avgs = []
        for region in regions_found:
            sals = regional_data[region]['salinities']
            if sals:
                sal_avgs.append(f"{sum(sals)/len(sals):.2f} PSU")
            else:
                sal_avgs.append("N/A")
        table_lines.append("| **Avg Salinity** | " + " | ".join(sal_avgs) + " |")

        response_parts.append("\n".join(table_lines))

        return "\n".join(response_parts)

    def _format_parametric_comparison(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format parameter-to-parameter comparison"""
        return self._format_generic_comparison(query, results)

    def _format_generic_comparison(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Generic comparison format when specific type can't be determined"""

        response_parts = []
        response_parts.append("## 📊 **Comparison Results**\n")
        response_parts.append(f"Found {len(results)} profiles for comparison\n")

        # Calculate overall statistics
        all_temps = []
        all_sals = []

        for row in results:
            if 'temperature' in row or 'surface_temperature' in row:
                temp = row.get('surface_temperature') or (
                    row['temperature'][0] if isinstance(row.get('temperature'), list) else None
                )
                if temp is not None:
                    all_temps.append(float(temp))

            if 'salinity' in row or 'surface_salinity' in row:
                sal = row.get('surface_salinity') or (
                    row['salinity'][0] if isinstance(row.get('salinity'), list) else None
                )
                if sal is not None:
                    all_sals.append(float(sal))

        if all_temps:
            response_parts.append(f"**Temperature:** {min(all_temps):.1f}°C to {max(all_temps):.1f}°C (avg: {sum(all_temps)/len(all_temps):.1f}°C)")

        if all_sals:
            response_parts.append(f"**Salinity:** {min(all_sals):.1f} to {max(all_sals):.1f} PSU (avg: {sum(all_sals)/len(all_sals):.1f} PSU)")

        return "\n".join(response_parts)

    def _infer_region(self, lat: float, lon: float) -> str:
        """Infer ocean region from coordinates"""

        # Indian Ocean
        if -90 <= lat <= 30 and 20 <= lon <= 120:
            if lat > 0 and 50 <= lon <= 80:
                return "Arabian Sea"
            elif lat > 0 and 80 <= lon <= 100:
                return "Bay of Bengal"
            else:
                return "Indian Ocean"

        # Pacific
        elif -90 <= lat <= 60 and (120 <= lon <= 180 or -180 <= lon <= -70):
            return "Pacific Ocean"

        # Atlantic
        elif -90 <= lat <= 70 and -70 <= lon <= 20:
            return "Atlantic Ocean"

        # Southern Ocean
        elif lat < -60:
            return "Southern Ocean"

        return "Unknown Region"


# Global instance
comparison_formatter = ComparisonFormatter()
