"""
AMFI Data Collector - PySpark Implementation

Collects mutual fund data from AMFI (Association of Mutual Funds in India) using PySpark.
AMFI provides official data including NAV, scheme details, and fund house information.
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, DateType, IntegerType
)
from pyspark.sql.window import Window
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMFICollector:
    """
    PySpark-based collector for AMFI mutual fund data with Parquet storage.
    
    AMFI provides official data for Indian mutual funds including:
    - Daily NAV data
    - Scheme master data
    - Fund house information
    
    Data is stored in Parquet format with date-based partitioning.
    """
    
    # AMFI URLs
    BASE_URL = "https://www.amfiindia.com/"
    NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
    HISTORICAL_NAV_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
    
    def __init__(self, app_name: str = "AMFICollector", data_path: str = "./data"):
        """
        Initialize AMFI collector with Spark session.
        
        Args:
            app_name: Spark application name
            data_path: Base path for storing Parquet files
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Data storage configuration
        self.data_path = data_path
        self.nav_data_path = f"{data_path}/nav_data"
        self.scheme_master_path = f"{data_path}/scheme_master"
        
        # HTTP session for data fetching
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def save_to_delta(self, df, path: str, partition_cols: List[str] = None, mode: str = "append"):
        """
        Save DataFrame to Delta format with optional partitioning.
        
        Args:
            df: Spark DataFrame to save
            path: Path to save the Delta table
            partition_cols: Columns to partition by (e.g., ['year', 'month'])
            mode: Write mode ('append', 'overwrite', 'ignore', 'error')
        """
        try:
            writer = df.write.format("delta").mode(mode)
            
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            
            # Enable schema evolution for new columns
            writer = writer.option("mergeSchema", "true")
            
            writer.save(path)
            logger.info(f"Successfully saved data to Delta table at {path}")
            
        except Exception as e:
            logger.error(f"Error saving data to Delta table at {path}: {e}")
            raise

    def read_from_delta(self, path: str, version: int = None, timestamp: str = None):
        """
        Read data from Delta table with optional time travel.
        
        Args:
            path: Path to the Delta table
            version: Specific version to read (time travel)
            timestamp: Specific timestamp to read (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            
        Returns:
            Spark DataFrame or None if table doesn't exist
        """
        try:
            reader = self.spark.read.format("delta")
            
            # Time travel options
            if version is not None:
                reader = reader.option("versionAsOf", version)
            elif timestamp is not None:
                reader = reader.option("timestampAsOf", timestamp)
            
            df = reader.load(path)
            logger.info(f"Successfully read data from Delta table at {path}")
            return df
            
        except Exception as e:
            logger.debug(f"Could not read Delta table from {path}: {e}")
            return None
        
    def get_daily_nav_data(self):
        """
        Get current day NAV data for all schemes from AMFI as Spark DataFrame.
        
        Returns:
            Spark DataFrame with columns: scheme_code, scheme_name, nav, date, fund_house, year, month
        """
        # Try to read from existing Delta table first
        today = datetime.now().strftime("%Y-%m-%d")
        existing_data = self._read_nav_data_by_date(today)
        
        if existing_data is not None and existing_data.count() > 0:
            logger.info(f"Found existing NAV data for {today}")
            return existing_data
        
        # Fetch fresh data from AMFI
        logger.info("Fetching fresh data from AMFI...")
        raw_data = self._fetch_raw_nav_data()
        nav_df = self._parse_nav_data_to_df(raw_data)
        
        if nav_df is not None:
            count = nav_df.count()
            logger.info(f"Successfully loaded {count} NAV records")
            
            # Always save fresh data to Delta table
            self._save_nav_data(nav_df)
            self._save_scheme_master(nav_df)
        
        return nav_df
    
    def _fetch_raw_nav_data(self) -> str:
        """
        Fetch raw NAV data from AMFI.
        
        Returns:
            Raw text content from AMFI NAV file
            
        Raises:
            requests.RequestException: If HTTP request fails
        """
        logger.info("Fetching daily NAV data from AMFI...")
        
        try:
            response = self.session.get(self.NAV_URL, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched {len(response.text)} characters from AMFI")
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"Error fetching daily NAV data: {e}")
            raise


    def _parse_nav_data_to_df(self, text_content: str):
        """
        Parse AMFI NAV text content and create Spark DataFrame.
        
        Args:
            text_content: Raw text from AMFI
            
        Returns:
            Spark DataFrame with parsed NAV data
        """
        # Save text content to temporary file for Spark to read
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True) as temp_file:
            temp_file.write(text_content)
            temp_file.flush()  # Ensure data is written to disk
            
            # Read as text file in Spark (while file still exists)
            raw_df = self.spark.read.text(temp_file.name)
            
            # Filter out empty lines and headers
            filtered_df = raw_df.filter(
                (f.col("value").isNotNull()) & 
                (f.col("value") != "") & 
                (~f.col("value").startswith("Scheme Code"))
            )
            
            # Separate fund house lines from scheme data lines
            scheme_data_df = filtered_df.filter(f.col("value").contains(";"))
            
            # Parse scheme data (format: Code;ISIN1;ISIN2;Name;NAV;Date)
            scheme_parsed = scheme_data_df.select(
                f.split(f.col("value"), ";").alias("parts")
            ).select(
                f.trim(f.col("parts")[0]).alias("scheme_code"),
                f.trim(f.col("parts")[3]).alias("scheme_name"),
                f.trim(f.col("parts")[4]).alias("nav_str"),
                f.trim(f.col("parts")[5]).alias("date_str")
            ).filter(
                # Filter out invalid records
                (f.col("scheme_code").isNotNull()) &
                (f.col("scheme_name").isNotNull()) &
                (f.col("nav_str").isNotNull()) &
                (f.col("date_str").isNotNull()) &
                (f.col("nav_str") != "N.A.") &
                (f.col("nav_str") != "-") &
                (f.col("nav_str") != "")
            )
            
            # Convert data types and add computed columns
            nav_df = scheme_parsed.select(
                f.col("scheme_code"),
                f.col("scheme_name"),
                f.col("nav_str").cast(DoubleType()).alias("nav"),
                f.to_date(f.col("date_str"), "dd-MMM-yyyy").alias("date")
            ).filter(
                f.col("nav").isNotNull() & 
                f.col("date").isNotNull()
            ).withColumn(
                "fund_house", f.lit("To be mapped")  # Placeholder for now
            ).withColumn(
                "year", f.year("date")
            ).withColumn(
                "month", f.month("date")
            )
            
            return nav_df
        # File is automatically deleted here when exiting the 'with' block

    def _read_nav_data_by_date(self, date_str: str):
        """
        Read NAV data for a specific date from Delta table.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Spark DataFrame or None if not found
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            year = target_date.year
            month = target_date.month
            
            # Try to read from partitioned Delta table
            # Delta will automatically use partition pruning for better performance
            df = self.read_from_delta(self.nav_data_path)
            
            if df is not None:
                # Filter by the specific date
                filtered_df = df.filter(f.col("date") == date_str)
                
                # Check if we have data for this date (much faster than count())
                if not filtered_df.isEmpty():
                    logger.info(f"Found data for date {date_str}")
                    return filtered_df
                
            logger.debug(f"No NAV data found for date {date_str}")
            return None
            
        except Exception as e:
            logger.debug(f"Could not read NAV data for {date_str}: {e}")
            return None
        
    def _save_nav_data(self, nav_df):
        """
        Save NAV data to Delta table with date-based partitioning.
        
        Args:
            nav_df: NAV DataFrame with columns including year, month for partitioning
        """
        try:
            # Ensure partitioning columns exist
            if "year" not in nav_df.columns:
                nav_df = nav_df.withColumn("year", f.year("date"))
            if "month" not in nav_df.columns:
                nav_df = nav_df.withColumn("month", f.month("date"))
            
            # Save to Delta table with partitioning
            self.save_to_delta(
                nav_df, 
                self.nav_data_path, 
                partition_cols=["year", "month"],
                mode="append"
            )
            
            logger.info("NAV data saved successfully to Delta table")
            
        except Exception as e:
            logger.error(f"Error saving NAV data: {e}")
            raise

    def _save_scheme_master(self, nav_df):
        """
        Save scheme master data (unique schemes) to Delta table.
        
        Args:
            nav_df: NAV DataFrame containing scheme information
        """
        try:
            # Window to get the latest record for each scheme
            window = Window.partitionBy("scheme_code").orderBy(f.col("date").desc())
            
            scheme_master = nav_df.select(
                "scheme_code", 
                "scheme_name", 
                "fund_house", 
                "date"
            ).withColumn(
                "row_num", f.row_number().over(window)
            ).filter(
                f.col("row_num") == 1  # Keep only the latest record per scheme
            ).drop("row_num", "date")  # Remove helper columns
            
            # Save to Delta table (overwrite to maintain clean master list)
            self.save_to_delta(
                scheme_master,
                self.scheme_master_path,
                mode="overwrite"
            )
            
            logger.info(f"Scheme master data saved successfully ({scheme_master.count()} unique schemes)")
            
        except Exception as e:
            logger.error(f"Error saving scheme master: {e}")
            raise

    def search_schemes(self, keyword: str, limit: int = 50):
        """
        Search for schemes by keyword in scheme name or fund house.
        
        Args:
            keyword: Search term (case-insensitive)
            limit: Maximum number of results to return
            
        Returns:
            Spark DataFrame with matching schemes
        """
        try:
            # Try to read from scheme master first (faster lookup)
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is not None:
                # Search in scheme master
                results = scheme_master.filter(
                    f.col("scheme_name").rlike(f"(?i).*{keyword}.*") |
                    f.col("fund_house").rlike(f"(?i).*{keyword}.*")
                ).limit(limit)
                
                if not results.isEmpty():
                    logger.info(f"Found schemes matching '{keyword}' in scheme master")
                    return results
            
            # Fallback to current NAV data if scheme master not available
            logger.info("Scheme master not available, searching in current NAV data")
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is not None:
                results = daily_nav.select(
                    "scheme_code", "scheme_name", "fund_house", "nav", "date"
                ).filter(
                    f.col("scheme_name").rlike(f"(?i).*{keyword}.*") |
                    f.col("fund_house").rlike(f"(?i).*{keyword}.*")
                ).limit(limit)
                
                return results
            
            # Return empty DataFrame if nothing found
            return self.spark.createDataFrame([], schema=StructType([]))
            
        except Exception as e:
            logger.error(f"Error searching schemes with keyword '{keyword}': {e}")
            return self.spark.createDataFrame([], schema=StructType([]))
        
    def get_scheme_details(self, scheme_code: str) -> Optional[Dict]:
        """
        Get detailed information for a specific scheme.
        
        Args:
            scheme_code: The scheme code to lookup
            
        Returns:
            Dictionary with scheme details or None if not found
        """
        try:
            # Get scheme info from master table
            scheme_info = None
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is not None:
                scheme_data = scheme_master.filter(f.col("scheme_code") == scheme_code)
                if not scheme_data.isEmpty():
                    scheme_info = scheme_data.first()
            
            # Get latest NAV data
            nav_info = None
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is not None:
                nav_data = daily_nav.filter(f.col("scheme_code") == scheme_code)
                if not nav_data.isEmpty():
                    nav_info = nav_data.first()
            
            # Combine information
            if scheme_info or nav_info:
                # Use scheme_info as primary, fallback to nav_info
                primary_source = scheme_info if scheme_info else nav_info
                
                details = {
                    'scheme_code': primary_source['scheme_code'],
                    'scheme_name': primary_source['scheme_name'],
                    'fund_house': primary_source['fund_house'],
                    'current_nav': nav_info['nav'] if nav_info else None,
                    'nav_date': nav_info['date'] if nav_info else None,
                    'data_source': 'scheme_master' if scheme_info else 'daily_nav'
                }
                
                logger.info(f"Found details for scheme {scheme_code}")
                return details
            
            logger.warning(f"Scheme code {scheme_code} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting scheme details for {scheme_code}: {e}")
            return None
        
    def close(self):
        """
        Close the Spark session and cleanup resources.
        
        This should be called when you're done using the collector
        to properly release Spark resources.
        """
        try:
            if hasattr(self, 'spark') and self.spark is not None:
                logger.info("Closing Spark session...")
                self.spark.stop()
                logger.info("Spark session closed successfully")
            
            # Close HTTP session if it exists
            if hasattr(self, 'session') and self.session is not None:
                self.session.close()
                logger.debug("HTTP session closed")
                
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
        
        finally:
            # Ensure references are cleared
            self.spark = None
            self.session = None

    def get_available_dates(self) -> List[str]:
        """
        Get list of dates for which NAV data is available.
        
        Returns:
            List of date strings in YYYY-MM-DD format, sorted chronologically
        """
        try:
            df = self.read_from_delta(self.nav_data_path)
            
            if df is None or df.isEmpty():
                logger.info("No NAV data found")
                return []
            
            # Get unique dates and sort them
            dates_df = df.select("date").distinct().orderBy("date")
            dates = [row["date"].strftime("%Y-%m-%d") for row in dates_df.collect()]
            
            logger.info(f"Found NAV data for {len(dates)} dates")
            return dates
            
        except Exception as e:
            logger.error(f"Error getting available dates: {e}")
            return []
        
    def get_nav_data_by_date_range(self, start_date: str, end_date: str):
        """
        Get NAV data for a specific date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)
            
        Returns:
            Spark DataFrame with NAV data for the date range
        """
        try:
            df = self.read_from_delta(self.nav_data_path)
            
            if df is None:
                logger.warning("No NAV data table found")
                return self.spark.createDataFrame([], schema=StructType([]))
            
            # Filter by date range
            filtered_df = df.filter(
                (f.col("date") >= start_date) & (f.col("date") <= end_date)
            ).orderBy("date", "scheme_code")
            
            if filtered_df.isEmpty():
                logger.info(f"No data found for date range {start_date} to {end_date}")
            else:
                logger.info(f"Retrieved NAV data for date range {start_date} to {end_date}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error getting NAV data for date range {start_date} to {end_date}: {e}")
            return self.spark.createDataFrame([], schema=StructType([]))
        
    def get_schemes_by_fund_house(self, fund_house: str):
        """
        Get all schemes for a specific fund house.
        
        Args:
            fund_house: Name of the fund house (case-insensitive)
            
        Returns:
            Spark DataFrame with schemes from the specified fund house
        """
        try:
            # Try scheme master first for better performance
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is not None:
                results = scheme_master.filter(
                    f.col("fund_house").rlike(f"(?i).*{fund_house}.*")
                ).orderBy("scheme_name")
                
                if not results.isEmpty():
                    logger.info(f"Found schemes for fund house '{fund_house}' from scheme master")
                    return results
            
            # Fallback to daily NAV data
            logger.info("Scheme master not available, searching in daily NAV data")
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is not None:
                results = daily_nav.select(
                    "scheme_code", "scheme_name", "fund_house", "nav", "date"
                ).filter(
                    f.col("fund_house").rlike(f"(?i).*{fund_house}.*")
                ).orderBy("scheme_name")
                
                return results
            
            return self.spark.createDataFrame([], schema=StructType([]))
            
        except Exception as e:
            logger.error(f"Error getting schemes for fund house '{fund_house}': {e}")
            return self.spark.createDataFrame([], schema=StructType([]))
        
    def get_fund_houses(self) -> List[str]:
        """
        Get list of all fund houses.
        
        Returns:
            List of fund house names, sorted alphabetically
        """
        try:
            # Try scheme master first (cleaner, deduplicated data)
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is not None:
                fund_houses_df = scheme_master.select("fund_house").distinct().orderBy("fund_house")
                fund_houses = [row["fund_house"] for row in fund_houses_df.collect()]
                
                if fund_houses:
                    logger.info(f"Found {len(fund_houses)} fund houses from scheme master")
                    return fund_houses
            
            # Fallback to daily NAV data
            logger.info("Scheme master not available, getting fund houses from daily NAV data")
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is not None:
                fund_houses_df = daily_nav.select("fund_house").distinct().orderBy("fund_house")
                fund_houses = [row["fund_house"] for row in fund_houses_df.collect()]
                
                logger.info(f"Found {len(fund_houses)} fund houses from daily NAV data")
                return fund_houses
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting fund houses: {e}")
            return []
        
    def get_nav_summary(self) -> Dict:
        """
        Get summary statistics of NAV data.
        
        Returns:
            Dictionary with summary statistics including counts, NAV ranges, dates
        """
        try:
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is None or daily_nav.isEmpty():
                return {'error': 'No NAV data available'}
            
            # Compute aggregations using Spark
            summary_stats = daily_nav.agg(
                f.count("*").alias("total_schemes"),
                f.min("nav").alias("min_nav"),
                f.max("nav").alias("max_nav"),
                f.avg("nav").alias("avg_nav"),
                f.countDistinct("fund_house").alias("total_fund_houses"),
                f.min("date").alias("earliest_date"),
                f.max("date").alias("latest_date")
            ).collect()[0]
            
            # Format the summary
            summary = {
                'total_schemes': summary_stats['total_schemes'],
                'total_fund_houses': summary_stats['total_fund_houses'],
                'nav_statistics': {
                    'min': round(summary_stats['min_nav'], 2) if summary_stats['min_nav'] else None,
                    'max': round(summary_stats['max_nav'], 2) if summary_stats['max_nav'] else None,
                    'average': round(summary_stats['avg_nav'], 2) if summary_stats['avg_nav'] else None
                },
                'date_range': {
                    'earliest': summary_stats['earliest_date'].strftime("%Y-%m-%d") if summary_stats['earliest_date'] else None,
                    'latest': summary_stats['latest_date'].strftime("%Y-%m-%d") if summary_stats['latest_date'] else None
                },
                'data_freshness': 'Current' if summary_stats['latest_date'] and 
                                (datetime.now().date() - summary_stats['latest_date']).days <= 1 else 'Stale'
            }
            
            logger.info(f"NAV summary generated: {summary['total_schemes']} schemes from {summary['total_fund_houses']} fund houses")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating NAV summary: {e}")
            return {'error': str(e)}
        
    def get_schemes_by_category(self, category_keyword: str):
        """
        Get schemes filtered by category keyword in scheme name.
        
        Args:
            category_keyword: Category keyword (e.g., 'Equity', 'Debt', 'Hybrid', 'ELSS')
            
        Returns:
            Spark DataFrame with schemes matching the category
        """
        try:
            # Try scheme master first for better performance
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is not None:
                results = scheme_master.filter(
                    f.col("scheme_name").rlike(f"(?i).*{category_keyword}.*")
                ).orderBy("scheme_name")
                
                if not results.isEmpty():
                    logger.info(f"Found schemes for category '{category_keyword}' from scheme master")
                    return results
            
            # Fallback to daily NAV data
            logger.info("Scheme master not available, searching in daily NAV data")
            daily_nav = self.get_daily_nav_data()
            
            if daily_nav is not None:
                results = daily_nav.select(
                    "scheme_code", "scheme_name", "fund_house", "nav", "date"
                ).filter(
                    f.col("scheme_name").rlike(f"(?i).*{category_keyword}.*")
                ).orderBy("scheme_name")
                
                return results
            
            return self.spark.createDataFrame([], schema=StructType([]))
            
        except Exception as e:
            logger.error(f"Error getting schemes for category '{category_keyword}': {e}")
            return self.spark.createDataFrame([], schema=StructType([]))
        
    def get_latest_nav_date(self) -> Optional[str]:
        """
        Get the most recent date for which NAV data is available.
        
        Returns:
            Latest date string in YYYY-MM-DD format or None if no data found
        """
        try:
            df = self.read_from_delta(self.nav_data_path)
            
            if df is None or df.isEmpty():
                logger.info("No NAV data found")
                return None
            
            # Get the maximum date
            latest_date_row = df.agg(f.max("date").alias("latest_date")).collect()[0]
            latest_date = latest_date_row["latest_date"]
            
            if latest_date:
                latest_date_str = latest_date.strftime("%Y-%m-%d")
                logger.info(f"Latest NAV data available for: {latest_date_str}")
                return latest_date_str
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest NAV date: {e}")
            return None
        
    def refresh_scheme_master(self, force_rebuild: bool = False):
        """
        Refresh the scheme master table from all available NAV data.
        
        Args:
            force_rebuild: If True, rebuilds from all historical data instead of just recent data
            
        Returns:
            Number of unique schemes processed
        """
        try:
            if force_rebuild:
                logger.info("Force rebuilding scheme master from all historical data...")
                # Use all available NAV data
                source_df = self.read_from_delta(self.nav_data_path)
            else:
                logger.info("Refreshing scheme master from recent data...")
                # Use current daily data (faster)
                source_df = self.get_daily_nav_data()
            
            if source_df is None or source_df.isEmpty():
                logger.warning("No source data available for scheme master refresh")
                return 0
            
            # Window to get the latest record for each scheme across all data
            window = Window.partitionBy("scheme_code").orderBy(f.col("date").desc())
            
            updated_scheme_master = source_df.select(
                "scheme_code", 
                "scheme_name", 
                "fund_house", 
                "date"
            ).withColumn(
                "row_num", f.row_number().over(window)
            ).filter(
                f.col("row_num") == 1  # Keep only the latest record per scheme
            ).drop("row_num", "date")  # Remove helper columns
            
            # Save to Delta table (overwrite to ensure clean master)
            self.save_to_delta(
                updated_scheme_master,
                self.scheme_master_path,
                mode="overwrite"
            )
            
            scheme_count = updated_scheme_master.count()
            logger.info(f"Scheme master refreshed successfully with {scheme_count} unique schemes")
            return scheme_count
            
        except Exception as e:
            logger.error(f"Error refreshing scheme master: {e}")
            raise

    def get_table_info(self, table_name: str = "nav_data") -> Dict:
        """
        Get Delta table metadata and information.
        
        Args:
            table_name: Table to inspect ('nav_data' or 'scheme_master')
            
        Returns:
            Dictionary with table metadata including versions, size, schema info
        """
        try:
            # Map table names to paths
            table_paths = {
                'nav_data': self.nav_data_path,
                'scheme_master': self.scheme_master_path
            }
            
            if table_name not in table_paths:
                return {'error': f"Unknown table: {table_name}. Use 'nav_data' or 'scheme_master'"}
            
            table_path = table_paths[table_name]
            
            # Check if table exists
            df = self.read_from_delta(table_path)
            if df is None:
                return {'error': f"Table '{table_name}' does not exist"}
            
            # Get basic table info
            row_count = df.count()
            column_count = len(df.columns)
            schema_info = [(field.name, str(field.dataType)) for field in df.schema.fields]
            
            # Get partition info (if partitioned)
            partition_columns = []
            try:
                # Try to get partition columns from the DataFrame
                if "year" in df.columns and "month" in df.columns:
                    partition_info = df.select("year", "month").distinct().collect()
                    partition_columns = [f"year={row['year']}/month={row['month']:02d}" for row in partition_info]
            except:
                pass
            
            # Try to get Delta table history (requires delta-spark)
            table_history = []
            try:
                # This would work with proper Delta Lake setup
                # history_df = self.spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`")
                # table_history = history_df.limit(5).collect()
                pass
            except:
                pass
            
            info = {
                'table_name': table_name,
                'table_path': table_path,
                'row_count': row_count,
                'column_count': column_count,
                'columns': [col_name for col_name, _ in schema_info],
                'schema': schema_info,
                'partitions': partition_columns[:10] if partition_columns else [],  # Limit to 10 for readability
                'partition_count': len(partition_columns),
                'table_format': 'Delta Lake',
                'status': 'Active'
            }
            
            logger.info(f"Retrieved info for table '{table_name}': {row_count} rows, {column_count} columns")
            return info
            
        except Exception as e:
            logger.error(f"Error getting table info for '{table_name}': {e}")
            return {'error': str(e)}
        
    def optimize_tables(self, table_name: str = "all", z_order_cols: List[str] = None) -> Dict:
        """
        Optimize Delta tables for better query performance.
        
        Args:
            table_name: Table to optimize ('nav_data', 'scheme_master', or 'all')
            z_order_cols: Columns to Z-order by for better performance
            
        Returns:
            Dictionary with optimization results
        """
        try:
            results = {}
            
            # Define tables to optimize
            tables_to_optimize = {}
            if table_name == "all":
                tables_to_optimize = {
                    'nav_data': self.nav_data_path,
                    'scheme_master': self.scheme_master_path
                }
            elif table_name == "nav_data":
                tables_to_optimize = {'nav_data': self.nav_data_path}
            elif table_name == "scheme_master":
                tables_to_optimize = {'scheme_master': self.scheme_master_path}
            else:
                return {'error': f"Unknown table: {table_name}. Use 'nav_data', 'scheme_master', or 'all'"}
            
            for tbl_name, tbl_path in tables_to_optimize.items():
                logger.info(f"Starting optimization for table: {tbl_name}")
                
                try:
                    # Check if table exists
                    if self.read_from_delta(tbl_path) is None:
                        results[tbl_name] = {'status': 'skipped', 'reason': 'table does not exist'}
                        continue
                    
                    # Run OPTIMIZE command
                    optimize_sql = f"OPTIMIZE delta.`{tbl_path}`"
                    
                    # Add Z-ordering if specified
                    if z_order_cols:
                        # Validate columns exist in the table
                        df = self.read_from_delta(tbl_path)
                        valid_cols = [col for col in z_order_cols if col in df.columns]
                        
                        if valid_cols:
                            optimize_sql += f" ZORDER BY ({', '.join(valid_cols)})"
                            logger.info(f"Z-ordering by columns: {valid_cols}")
                        else:
                            logger.warning(f"None of the Z-order columns {z_order_cols} exist in {tbl_name}")
                    elif tbl_name == "nav_data":
                        # Default Z-order for nav_data
                        optimize_sql += " ZORDER BY (date, scheme_code)"
                        logger.info("Using default Z-order: date, scheme_code")
                    
                    # Execute optimization
                    optimize_result = self.spark.sql(optimize_sql)
                    optimize_metrics = optimize_result.collect()
                    
                    # Run VACUUM to clean up old files (keep 7 days retention)
                    vacuum_sql = f"VACUUM delta.`{tbl_path}` RETAIN 168 HOURS"  # 7 days
                    logger.info(f"Running VACUUM on {tbl_name}...")
                    self.spark.sql(vacuum_sql)
                    
                    results[tbl_name] = {
                        'status': 'success',
                        'optimize_metrics': str(optimize_metrics[0]) if optimize_metrics else 'completed',
                        'operations': ['OPTIMIZE', 'VACUUM'],
                        'z_order_columns': valid_cols if z_order_cols else (['date', 'scheme_code'] if tbl_name == 'nav_data' else [])
                    }
                    
                    logger.info(f"Successfully optimized table: {tbl_name}")
                    
                except Exception as table_error:
                    results[tbl_name] = {
                        'status': 'failed',
                        'error': str(table_error)
                    }
                    logger.error(f"Failed to optimize table {tbl_name}: {table_error}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during table optimization: {e}")
            return {'error': str(e)}
        
    def validate_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality checks on NAV data.
        
        Returns:
            Dictionary with validation results and quality metrics
        """
        try:
            logger.info("Starting data quality validation...")
            
            # Get the data to validate
            df = self.read_from_delta(self.nav_data_path)
            if df is None or df.isEmpty():
                return {'status': 'failed', 'reason': 'No data available for validation'}
            
            validation_results = {}
            
            # 1. Check for null/missing values
            null_checks = {}
            for col_name in ['scheme_code', 'scheme_name', 'nav', 'date']:
                if col_name in df.columns:
                    null_count = df.filter(f.col(col_name).isNull()).count()
                    total_count = df.count()
                    null_checks[col_name] = {
                        'null_count': null_count,
                        'null_percentage': round((null_count / total_count) * 100, 2) if total_count > 0 else 0
                    }
            
            validation_results['null_checks'] = null_checks
            
            # 2. Check for invalid NAV values
            invalid_nav_count = df.filter((f.col("nav") <= 0) | (f.col("nav") > 10000)).count()
            validation_results['invalid_nav_values'] = {
                'count': invalid_nav_count,
                'percentage': round((invalid_nav_count / df.count()) * 100, 2) if df.count() > 0 else 0
            }
            
            # 3. Check for duplicate scheme codes on same date
            duplicate_check = df.groupBy("scheme_code", "date").agg(
                f.count("*").alias("record_count")
            ).filter(f.col("record_count") > 1)
            
            duplicate_count = duplicate_check.count()
            validation_results['duplicate_records'] = {
                'count': duplicate_count,
                'sample_duplicates': [row.asDict() for row in duplicate_check.limit(5).collect()] if duplicate_count > 0 else []
            }
            
            # 4. Check date consistency
            date_stats = df.agg(
                f.min("date").alias("min_date"),
                f.max("date").alias("max_date"),
                f.countDistinct("date").alias("unique_dates")
            ).collect()[0]
            
            validation_results['date_consistency'] = {
                'date_range': {
                    'earliest': date_stats['min_date'].strftime("%Y-%m-%d") if date_stats['min_date'] else None,
                    'latest': date_stats['latest_date'].strftime("%Y-%m-%d") if date_stats['max_date'] else None
                },
                'unique_dates': date_stats['unique_dates']
            }
            
            # 5. Check scheme code patterns
            invalid_scheme_codes = df.filter(
                ~f.col("scheme_code").rlike("^[0-9]+$")  # Should be numeric
            ).count()
            
            validation_results['scheme_code_validation'] = {
                'invalid_format_count': invalid_scheme_codes,
                'percentage': round((invalid_scheme_codes / df.count()) * 100, 2) if df.count() > 0 else 0
            }
            
            # 6. Overall quality score
            total_issues = (
                sum([check['null_count'] for check in null_checks.values()]) +
                invalid_nav_count +
                duplicate_count +
                invalid_scheme_codes
            )
            
            total_records = df.count()
            quality_score = max(0, round(100 - (total_issues / total_records) * 100, 2)) if total_records > 0 else 0
            
            validation_results['overall_quality'] = {
                'score': quality_score,
                'total_records': total_records,
                'total_issues': total_issues,
                'status': 'excellent' if quality_score >= 95 else 'good' if quality_score >= 85 else 'needs_attention'
            }
            
            logger.info(f"Data quality validation completed. Quality score: {quality_score}%")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during data quality validation: {e}")
            return {'status': 'error', 'message': str(e)}
        
    def check_duplicate_schemes(self) -> Dict:
        """
        Check for potential duplicate schemes based on similar names or codes.
        
        Returns:
            Dictionary with duplicate scheme analysis
        """
        try:
            logger.info("Starting duplicate scheme analysis...")
            
            # Get scheme master data (cleaner for analysis)
            scheme_master = self.read_from_delta(self.scheme_master_path)
            
            if scheme_master is None:
                # Fallback to daily NAV data
                daily_nav = self.get_daily_nav_data()
                if daily_nav is None or daily_nav.isEmpty():
                    return {'status': 'failed', 'reason': 'No data available for duplicate analysis'}
                
                # Create temporary scheme data
                scheme_master = daily_nav.select("scheme_code", "scheme_name", "fund_house").distinct()
            
            if scheme_master.isEmpty():
                return {'status': 'failed', 'reason': 'No scheme data found'}
            
            duplicate_analysis = {}
            
            # 1. Check for duplicate scheme codes
            code_duplicates = scheme_master.groupBy("scheme_code").agg(
                f.count("*").alias("count"),
                f.collect_list("scheme_name").alias("scheme_names")
            ).filter(f.col("count") > 1)
            
            code_duplicate_list = []
            for row in code_duplicates.collect():
                code_duplicate_list.append({
                    'scheme_code': row['scheme_code'],
                    'count': row['count'],
                    'scheme_names': row['scheme_names']
                })
            
            duplicate_analysis['duplicate_codes'] = {
                'count': len(code_duplicate_list),
                'details': code_duplicate_list
            }
            
            # 2. Check for very similar scheme names (within same fund house)
            # This is a simplified check - in production you'd use more sophisticated string matching
            similar_names = []
            
            # Group by fund house and look for similar names
            fund_house_groups = scheme_master.groupBy("fund_house").agg(
                f.collect_list(f.struct("scheme_code", "scheme_name")).alias("schemes")
            ).collect()
            
            for fund_house_row in fund_house_groups:
                fund_house = fund_house_row['fund_house']
                schemes = fund_house_row['schemes']
                
                # Compare scheme names within the same fund house
                for i, scheme1 in enumerate(schemes):
                    for j, scheme2 in enumerate(schemes[i+1:], i+1):
                        name1 = scheme1['scheme_name'].lower().replace(' ', '').replace('-', '')
                        name2 = scheme2['scheme_name'].lower().replace(' ', '').replace('-', '')
                        
                        # Simple similarity check (can be improved with Levenshtein distance)
                        if name1 in name2 or name2 in name1:
                            if abs(len(name1) - len(name2)) <= 3:  # Similar lengths
                                similar_names.append({
                                    'fund_house': fund_house,
                                    'scheme1': {
                                        'code': scheme1['scheme_code'],
                                        'name': scheme1['scheme_name']
                                    },
                                    'scheme2': {
                                        'code': scheme2['scheme_code'], 
                                        'name': scheme2['scheme_name']
                                    }
                                })
                                
                        # Limit to prevent too many results
                        if len(similar_names) >= 20:
                            break
                    if len(similar_names) >= 20:
                        break
                if len(similar_names) >= 20:
                    break
            
            duplicate_analysis['similar_names'] = {
                'count': len(similar_names),
                'details': similar_names
            }
            
            # 3. Check for schemes with identical names but different codes
            name_duplicates = scheme_master.groupBy("scheme_name").agg(
                f.count("*").alias("count"),
                f.collect_list("scheme_code").alias("scheme_codes"),
                f.first("fund_house").alias("fund_house")
            ).filter(f.col("count") > 1)
            
            name_duplicate_list = []
            for row in name_duplicates.collect():
                name_duplicate_list.append({
                    'scheme_name': row['scheme_name'],
                    'count': row['count'],
                    'scheme_codes': row['scheme_codes'],
                    'fund_house': row['fund_house']
                })
            
            duplicate_analysis['duplicate_names'] = {
                'count': len(name_duplicate_list),
                'details': name_duplicate_list
            }
            
            # 4. Summary
            total_duplicates = (
                duplicate_analysis['duplicate_codes']['count'] +
                duplicate_analysis['similar_names']['count'] +
                duplicate_analysis['duplicate_names']['count']
            )
            
            duplicate_analysis['summary'] = {
                'total_potential_duplicates': total_duplicates,
                'total_schemes_analyzed': scheme_master.count(),
                'duplicate_percentage': round((total_duplicates / scheme_master.count()) * 100, 2) if scheme_master.count() > 0 else 0,
                'status': 'clean' if total_duplicates == 0 else 'needs_review' if total_duplicates <= 10 else 'significant_issues'
            }
            
            logger.info(f"Duplicate analysis completed. Found {total_duplicates} potential duplicate issues")
            return duplicate_analysis
            
        except Exception as e:
            logger.error(f"Error during duplicate scheme analysis: {e}")
            return {'status': 'error', 'message': str(e)}